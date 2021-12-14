# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""

import random
import pickle
import logging
import torch
import torch.nn.functional as F
from torch import nn
import math

from deformable_potr_plus.util import box_ops
from deformable_potr_plus.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from deformable_potr_plus.models.backbone import build_backbone
from deformable_potr_plus.models.matcher import build_matcher
from deformable_potr_plus.models.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from deformable_potr_plus.models.deformable_transformer import build_deformable_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformablePOTR(nn.Module):
    """ Deformable POTR module for joint coordinates regression """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False,
                 two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer

        hidden_dim = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 3, 1)
        self.num_feature_levels = num_feature_levels

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))

            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)

        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # TODO: Is this necessary?
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        # END OF TODO

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed

        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs, masks = [], []

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        return {"pred_logits": outputs_class[-1], "pred_coords": outputs_coord[-1]}

    def convert_outputs_average_decoding(self, model_outputs):
        """
        Decodes the resulting coordinates by averaging all queries which predicted (claimed) to be of the given class.
        
        :param model_outputs: Outputs from the DeformablePOTR model
        :return: Predicted averaged coordinates tensor (batch, num_class, num_dim)
        """""

        # Convert outputs to averaged class joints
        pred_class = torch.argmax(model_outputs["pred_logits"], dim=2)
        all_pred_coords = [[[] for _ in range(14)] for _ in range(len(model_outputs["pred_logits"]))]

        for target_i in range(len(model_outputs["pred_logits"])):
            for i, c in enumerate(pred_class[target_i, :]):
                if c.item() != 14:
                    all_pred_coords[target_i][c.item()].append(model_outputs["pred_coords"][target_i, i])

        all_pred_coords = [[torch.stack(all_pred_coords[target_i][joint_i]) if all_pred_coords[target_i][joint_i] else torch.zeros(1, 3) for joint_i in range(14)] for target_i in range(len(outputs["pred_logits"]))]
        avg_pred_coords = [[torch.Tensor([torch.mean(joint_batch[:, 0]), torch.mean(joint_batch[:, 1]), torch.mean(joint_batch[:, 2])]) for joint_batch in target_batch] for target_batch in all_pred_coords]
        avg_output_coords = torch.stack([torch.stack(target_batch) for target_batch in avg_pred_coords])

        return avg_output_coords


class SetCriterion(nn.Module):
    """ This class computes the loss for POTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, cube_size=300):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            cube_size: the milimeter size of the cube in which the hand coordinates are located
        """

        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.cube_size = cube_size

    def loss_labels(self, outputs, targets, indices):

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o.cuda()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, self.num_classes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_coords(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        idx = self._get_src_permutation_idx(indices)
        src_coords = outputs["pred_coords"][idx]
        target_coords = torch.cat([t["coords"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_coords = F.smooth_l1_loss(src_coords, target_coords, reduction="none")
        losses = {}
        losses['loss_coords'] = loss_coords.sum() / self.num_classes

        return losses

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "coords": self.loss_coords
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        targets = [{
            "coords": target,
            "labels": torch.tensor(list(range(14)))
        } for target in targets]

        indices = self.matcher(outputs, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        logging.info(str(losses))

        return losses

    def get_avg_L2_prediction_error_hungarian_matcher_decoding(self, outputs, targets):
        """
        Decodes the outputs using Hungarian matcher and calculates the average L2 distance (prediction error) per batch
        per single joint.

        :param outputs:
        :param targets:
        :return: Average prediction error (in millimeters)
        """

        targets = [{
            "coords": target,
            "labels": torch.tensor(list(range(14)))
        } for target in targets]

        indices = self.matcher(outputs, targets)

        idx = self._get_src_permutation_idx(indices)
        src_coords = outputs["pred_coords"][idx]
        target_coords = torch.cat([t["coords"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        return self.calculate_avg_L2_distance(src_coords, target_coords)

    def calculate_avg_L2_distance(self, output_coords, target_coords):
        """
        Calculates the average L2 distance (prediction error) per batch per single joint.

        :param output_coords:
        :param target_coords:
        :return: Average prediction error (in millimeters)
        """

        try:
            # Obtain the average L2 distance per a single joint estimation (therefore the overall result is divided by
            # the number of targets and classes) and convert this relative distance form (as the dataset target values
            # are on the [-1; 1] range to absolute mms
            loss_coords = F.mse_loss(output_coords, target_coords, reduction="none")
            res = (math.sqrt(float(torch.sum(loss_coords))) / len(target_coords) / self.num_classes) * (self.cube_size * 2)

        except:
            logging.warning("A problem ocurred during L2 error calculation!")
            return 0

        return res


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    device = torch.device(args.device)
    backbone = build_backbone(args)
    matcher = build_matcher(args)
    transformer = build_deformable_transformer(args)

    model = DeformablePOTR(
        backbone,
        transformer,
        num_classes=args.num_classes + 1,  # Add one additional class for representation of the background
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )

    weight_dict = {"loss_coords": 1, "labels": 1}
    losses = ['coords', "labels"]

    criterion = SetCriterion(args.num_classes, matcher, weight_dict=weight_dict, losses=losses, cube_size=args.cube_size)
    criterion.to(device)

    return model, criterion

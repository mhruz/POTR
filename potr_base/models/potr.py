
import torch
import torch.nn.functional as F
from torch import nn

from potr_base.util.misc import NestedTensor, nested_tensor_from_tensor_list
from potr_base.models.backbone import build_backbone
from potr_base.models.transformer import build_transformer


class POTR(nn.Module):
    """
    The POTR (DEtection TRansformer) module, inspired by the "End to End Object Detection Using Transformers" paper.
    """

    def __init__(self, backbone, transformer, num_queries):
        """
        Initializes the POTR model.

        :param backbone: Backbone (convolutional neural network) for extraction of compact feature representation to be
            used
        :param transformer: Transformer Encoder-Decoder model to be used
        :param num_classes: Number of object classes
        :param num_queries: Number of object queries (for the Transformer Decoder)
        :param aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used
        """

        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer

        hidden_dim = transformer.d_model

        self.coords_embed = MLP(hidden_dim, hidden_dim, 3, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor):
        """
        :param samples: A NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        :return: A dictionary with the following elements:
            - "pred_logits": The classification logits (including no-object) for all queries. Shape = [batch_size *
                    num_queries * (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as (center_x, center_y,
                    height, width). These values are normalized in [0, 1], relative to the size of each individual image
                    (disregarding possible padding).
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of dictionnaries
                    containing the two above keys for each decoder layer.
        """

        # Convert the input to the NestedTensor, if necessary
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Propagate the inputs through the backbone CNN
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()

        # Ensure that outputted mask is valid
        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # outputs_coord = self.coords_embed(hs).sigmoid()
        outputs_coord = self.coords_embed(hs)
        return outputs_coord[-1]


class SetCriterion(nn.Module):
    """
    Module for computation of the loss for POTR. The process consists of two steps:
        - computation of Hungarian assignment between ground truth boxes and the outputs of the model
        - supervision of each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_queries, weight_dict, losses):
        """
        Creates the criterion.

        :param num_queries: Number of query slots (landmarks to be regressed).
        :param weight_dict: Dictionary containing as key the names of the losses and as values their relative weight.
        :param losses: List of all the losses to be applied. See get_loss for list of available losses.
        """

        super().__init__()
        self.num_queries = num_queries
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_coords(self, outputs, targets):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        loss_coords = F.smooth_l1_loss(outputs, torch.stack(targets), reduction="none")

        losses = {}
        losses['loss_coords'] = loss_coords.sum() / self.num_queries

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'coords': self.loss_coords
        }
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def build(args):

    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = POTR(
        backbone,
        transformer,
        num_queries=args.num_queries
    )

    weight_dict = {"loss_coords": 1}

    losses = ['coords']
    criterion = SetCriterion(args.num_queries, weight_dict=weight_dict, losses=losses)
    criterion.to(device)

    return model, criterion

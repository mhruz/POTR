
"""
Standalone annotations script for hand pose estimation using DETR HPOES.

! Warning !
The script expects the input data in a .H5 dataset (under "images" key) in a form specified below. For other data
structures, please implement your own logic. Either way, the `depth_maps` should be an array of size (N, 256, 256).
"""

import argparse
import torch
import io
import h5py

import numpy as np

from torch.utils.data import DataLoader

from deformable_potr.models.deformable_potr import DeformablePOTR
from deformable_potr.models.backbone import build_backbone
from deformable_potr.models.deformable_transformer import build_deformable_transformer

from dataset.dataset_processing import aug_morph_close
from dataset.hpoes_dataset import HPOESOberwegerDataset


# Arguments
parser = argparse.ArgumentParser("DETR HPOES Standalone Annotations Script", add_help=False)
parser.add_argument("--weights_file", type=str, default="out/checkpoint.pth",
                    help="Path to the pretrained model's chekpoint (.pth)")
parser.add_argument("--input_file", type=str, default="in.h5",
                    help="Path to the .h5 file with input data (depth maps)")
parser.add_argument("--output_file", type=str, default="out.h5", help="Path to the .h5 file to write into")
parser.add_argument("--device", default="cpu", help="Device to be used")
parser.add_argument("--tta", default=1, help="Whether to use Test Time Augmentation")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
args = parser.parse_args()

device = torch.device(args.device)

# Load the input data and checkpoint
print("Loading the input data and checkpoints.")
# input_datafile = h5py.File(args.input_file, "r")
output_datafile = h5py.File(args.output_file, 'w')
checkpoint = torch.load(args.weights_file, map_location=device)

output_list = []

# Construct the model from the loaded data
model = DeformablePOTR(
    build_backbone(checkpoint["args"]),
    build_deformable_transformer(checkpoint["args"]),
    num_queries=checkpoint["args"].num_queries,
    num_feature_levels=checkpoint["args"].num_feature_levels
)
model.load_state_dict(checkpoint["model"])
model.to(device)

print("Constructed model successfully.")

dataset_test = HPOESOberwegerDataset(args.input_file, encoded=True, mode='test')
data_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=True)

# Iterate over the depth maps and structure the predictions
for i, (samples) in enumerate(data_loader):
    print(i)
    samples = samples.to(device, dtype=torch.float32)

    results = model(samples).detach().cpu().numpy()
    output_list.append(results)

print("Predictions were made.")

output = np.asarray(output_list).squeeze()
output_datafile.create_dataset("estimated_hands", data=output)
output_datafile.close()

print("Data was successfully structured and saved.")

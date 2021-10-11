
"""
Standalone annotations script for hand pose estimation using POTR HPOES.

! Warning !
The script expects the input data in a .H5 dataset (under "images" key) in a form specified below. For other data
structures, please implement your own logic. Either way, the `depth_maps` should be an array of size (N, 256, 256).
"""

import argparse
import torch
import h5py

import numpy as np

from torch.utils.data import DataLoader

from potr_base.models.potr import POTR
from potr_base.models.backbone import build_backbone
from potr_base.models.transformer import build_transformer

from dataset.hpoes_dataset import HPOESOberwegerDataset


# Arguments
parser = argparse.ArgumentParser("POTR HPOES Standalone Annotations Script", add_help=False)
parser.add_argument("--weights_file", type=str, default="out/checkpoint.pth",
                    help="Path to the pretrained model's chekpoint (.pth)")
parser.add_argument("--input_file", type=str, default="in.h5",
                    help="Path to the .h5 file with input data (depth maps)")
parser.add_argument("--output_file", type=str, default="out.h5", help="Path to the .h5 file to write into")
parser.add_argument("--device", default="cpu", help="Device to be used")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
args = parser.parse_args()

device = torch.device(args.device)

# Load the input data and checkpoint
print("Loading the input data and checkpoints.")
output_datafile = h5py.File(args.output_file, 'w')
checkpoint = torch.load(args.weights_file, map_location=device)

output_list = []

# Construct the model from the loaded data
model = POTR(
    build_backbone(checkpoint["args"]),
    build_transformer(checkpoint["args"]),
    num_queries=checkpoint["args"].num_queries
)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)

print("Constructed model successfully.")

dataset_test = HPOESOberwegerDataset(args.input_file, encoded=True, mode='test')
data_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=False)

# Iterate over the depth maps and structure the predictions
for i, (samples) in enumerate(data_loader):
    print(i)
    samples = samples.to(device, dtype=torch.float32)

    results = model(samples).detach().cpu().numpy()
    output_list.extend(results)

print("Predictions were made.")

output = np.asarray(output_list).squeeze()
output_datafile.create_dataset("estimated_hands", data=output)
output_datafile.close()

print("Data was successfully structured and saved.")

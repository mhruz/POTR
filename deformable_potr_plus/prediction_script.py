
"""
Standalone annotations script for hand pose estimation using DETR HPOES.

! Warning !
The script expects the input data in a .H5 dataset (under "images" key) in a form specified below. For other data
structures, please implement your own logic. Either way, the `depth_maps` should be an array of size (N, 256, 256).
"""

import argparse
import torch
import h5py

import numpy as np

from torch.utils.data import DataLoader

from deformable_potr_plus.models.deformable_potr_plus import DeformablePOTR
from deformable_potr_plus.models.backbone import build_backbone
from deformable_potr_plus.models.deformable_transformer import build_deformable_transformer
from deformable_potr_plus.models.matcher_prediction import build_matcher

from dataset.hpoes_dataset import HPOESOberwegerDataset


# Arguments
parser = argparse.ArgumentParser("DETR HPOES Standalone Annotations Script", add_help=False)
parser.add_argument("--weights_file", type=str, default="out/checkpoint.pth",
                    help="Path to the pretrained model's chekpoint (.pth)")
parser.add_argument("--input_file", type=str, default="in.h5",
                    help="Path to the .h5 file with input data (depth maps)")
parser.add_argument("--output_file", type=str, default="out.h5", help="Path to the .h5 file to write into")
parser.add_argument("--device", default="cpu", help="Device to be used")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_coord', default=5, type=float, help="Coordinate distance coefficient in the matching cost")
parser.add_argument('--data_resolution', default=(224, 224), type=int, nargs=2, help="Resolution of the input data.")
args = parser.parse_args()

device = torch.device(args.device)

# Load the input data and checkpoint
print("Loading the input data and checkpoints.")
output_datafile = h5py.File(args.output_file, 'w')
checkpoint = torch.load(args.weights_file, map_location=device)

output_list = []

# Construct the model from the loaded data
backbone = build_backbone(checkpoint["args"])
matcher_prediction = build_matcher()
transformer = build_deformable_transformer(checkpoint["args"])
model = DeformablePOTR(
        backbone,
        transformer,
        num_classes=checkpoint["args"].num_classes + 1,  # Add one additional class for representation of the background
        num_queries=checkpoint["args"].num_queries,
        num_feature_levels=checkpoint["args"].num_feature_levels,
    )

model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)

print("Constructed model successfully.")

dataset_test = HPOESOberwegerDataset(args.input_file, data_resolution=args.data_resolution, encoded=True, mode='eval')
data_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=False)

# Iterate over the depth maps and structure the predictions
for i, (samples, targets, cubes) in enumerate(data_loader):
    print(i)
    samples = [item.to(device, dtype=torch.float32) for item in samples]
    targets = [item.to(device) for item in targets]
    '''targets = [{
            "coords": target.to(device),
            "labels": torch.tensor(list(range(target.shape[0]))).to(device)
        } for target in targets]'''

    results = model(samples)

    out = matcher_prediction(results)

    output_list.extend(out)

print("Predictions were made.")

output = np.asarray(output_list).squeeze()
output_datafile.create_dataset("estimated_hands", data=output)
output_datafile.close()

print("Data was successfully structured and saved.")

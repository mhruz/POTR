import math
import torch
import logging
import random
import io

import numpy as np
import torch.utils.data as torch_data

from dataset.dataset_processing import load_hpoes_data, load_encoded_hpoes_data

import cv2
import matplotlib.pyplot as plt


def vis_keypoints(image_orig, keypoints, diameter=2):
    image = image_orig.copy()
    image += 1
    image *= 127.5
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image, cmap="Purples")
    plt.show()


class HPOESDataset:
    """Object representation of the HPOES dataset for loading hand joints landmarks"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, data: [np.ndarray], labels: [np.ndarray]):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param data: List of numpy.ndarrays with the depth maps
        :param labels: List of numpy.ndarrays with the 3D coordinates of the individual hand joints
        """

        # Prevent from initiating with invalid data
        if len(data) != len(labels):
            logging.error("The size of the data (depth maps) list is not equal to the size of the labels list.")

        self.data = data
        self.labels = labels

    def generate_random_subsets(self, batch_size: int) -> [[dict]]:
        """
        Converts the data into PyTorch Tensors and generates a fully random list of structured subsets (batches) of the
        desired size.

        :param batch_size: Size of each individual batch
        :return: List of list of dictionaries with elements:
            - data: Torch.Tensor with the depth maps (224x224)
            - labels: LTorch.Tensor with the 3D coordinates of the individual hand joints (in millimeters, relative to
            the center of the image) (21x3)
        """

        output = []
        num_batches = math.ceil(len(self.labels) / batch_size)

        # Batch together the data with the corresponding labels in order to shuffle them
        batched_data = [(data, labels) for (data, labels) in zip(self.data, self.labels)]
        random.shuffle(batched_data)

        # Split the data based on the given parameters
        split_batched_data = np.array_split(batched_data, num_batches)

        # Convert the data into lists of dictionaries with PyTorch Tensors
        for data_batch in split_batched_data:
            output.append([{"data": torch.from_numpy(data), "labels": torch.from_numpy(labels)} for (data, labels) in
                           data_batch])

        return output

    def generate_continuous_subsets(self, batch_size: int) -> [[dict]]:
        """
        Converts the data into PyTorch Tensors and generates a list of structured subsets (batches) of the desired size.

        :param batch_size: Size of each individual batch
        :return: List of list of dictionaries with elements:
            - data: Torch.Tensor with the depth maps (224x224)
            - labels: LTorch.Tensor with the 3D coordinates of the individual hand joints (in millimeters, relative to
            the center of the image) (21x3)
        """

        output = []
        num_batches = math.ceil(len(self.labels) / batch_size)

        # Split the data based on the given parameters
        split_data, split_labels = np.array_split(self.data, num_batches), np.array_split(self.labels, num_batches)

        # Convert the data into lists of dictionaries with PyTorch Tensors
        for data_batch, labels_batch in zip(split_data, split_labels):
            output.append([{"data": torch.from_numpy(data), "labels": torch.from_numpy(labels)} for (data, labels) in
                           zip(data_batch, labels_batch)])

        return output


class HPOESAdvancedDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, transform=None):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        loaded_data = load_hpoes_data(dataset_filename)
        data, labels = loaded_data["data"], loaded_data["labels"]

        # Prevent from initiating with invalid data
        if len(data) != len(labels):
            logging.error("The size of the data (depth maps) list is not equal to the size of the labels list.")

        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        depth_map = torch.from_numpy(self.data[idx]) / 224.0 - 0.5
        label = torch.from_numpy(self.labels[idx]) / 150.0

        # Perform any additionally desired transformations
        if self.transform:
            depth_map, label = self.transform(depth_map, label)

        return depth_map, label

    def __len__(self):
        return len(self.labels)


class HPOESOberwegerDataset(torch_data.Dataset):
    """Object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties. The data are in format defined by Oberweger: 224x224 depth images (-1, 1)
    with labels -1.0 to +1.0 from cube = (250, 250, 250)"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, encoded=True, transform=None):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        :param encoded: Whether to read only encoded data and decode them at runtime (default: True)
        """
        self.encoded = encoded

        if not encoded:
            loaded_data = load_hpoes_data(dataset_filename)
        else:
            loaded_data = load_encoded_hpoes_data(dataset_filename)

        data, labels = loaded_data["data"], loaded_data["labels"]

        # Prevent from initiating with invalid data
        if len(data) != len(labels):
            logging.error("The size of the data (depth maps) list is not equal to the size of the labels list.")

        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        if self.encoded:
            _file = io.BytesIO(self.data[idx])
            depth_map = np.load(_file)["arr_0"]

        else:
            depth_map = self.data[idx]

        label = self.labels[idx]

        # Perform any additionally desired transformations
        if self.transform:
            # transform the labels into image coordinate
            label = depth_map.shape[0] // 2 * label + depth_map.shape[0] // 2
            keypoints = label[:, :2].tolist()

            transformed = self.transform(image=depth_map, keypoints=keypoints)
            depth_map = transformed["image"]
            keypoints = transformed["keypoints"]

            keypoints = np.asarray(keypoints)
            # vis_keypoints(depth_map, keypoints[:, :2])

            label[:, 0] = keypoints[:, 0]
            label[:, 1] = keypoints[:, 1]

            label = (label - depth_map.shape[0] // 2) / (depth_map.shape[0] // 2)

        depth_map = torch.from_numpy(depth_map)
        label = torch.from_numpy(np.asarray(label))

        return depth_map, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    pass

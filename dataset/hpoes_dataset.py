import math
import torch
import logging
import random
import io

import numpy as np
import torch.utils.data as torch_data

from dataset.dataset_processing import load_hpoes_data, load_encoded_hpoes_data, aug_translate_depth, \
    aug_morph_close, aug_keypoints

import cv2
import matplotlib.pyplot as plt

import albumentations as A


def vis_keypoints(image_orig, keypoints, diameter=2, show=True):
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
    if show:
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

    def __init__(self, dataset_filename: str, data_resolution: int, encoded=True,
                 transform=None, p_augment_3d=0.0, mode: str = 'test'):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param data_resolution: resolution of the input data
        :param transform: Any data transformation to be applied (default: None)
        :param encoded: Whether to read only encoded data and decode them at runtime (default: True)
        :param mode: train, eval, test
        """
        self.data_resolution = data_resolution
        self.encoded = encoded
        self.p_augment_3d = p_augment_3d

        if not encoded:
            loaded_data = load_hpoes_data(dataset_filename, mode=mode)
        else:
            loaded_data = load_encoded_hpoes_data(dataset_filename, mode=mode)

        data = loaded_data["data"]
        if mode == 'test':
            labels = None
        else:
            labels = loaded_data["labels"]
            # Prevent from initiating with invalid data
            if len(data) != len(labels):
                logging.error("The size of the data (depth maps) list is not equal to the size of the labels list.")

        self.data = data
        self.labels = labels
        self.transform = transform
        self.mode = mode

    @staticmethod
    def preprocessing():
        transform = A.Compose([
            A.Lambda(image=aug_morph_close, keypoint=aug_keypoints, p=1.0),
        ])
        return transform

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

        if self.labels:
            label = self.labels[idx]
        else:
            label = None

        if self.mode == 'test' or self.mode == 'eval':
            preprocessing = self.preprocessing()
            transformed = preprocessing(image=depth_map)
            depth_map = transformed["image"]
            depth_map = torch.from_numpy(depth_map)
            depth_map = depth_map.unsqueeze(0).expand(3, self.data_resolution, self.data_resolution)
            if label is not None:
                label = torch.from_numpy(np.asarray(label))
                return depth_map, label
            return depth_map

        # Perform any additionally desired transformations
        if self.transform:
            # transform the labels into image coordinate
            # the labels are expected to be in relative coordinates of the volume they stem from
            # with (-1, -1, -1) in the Top-Left-Front corner of the volume
            # (0, 0, 0) in the center and (1, 1, 1) in the Bottom-Right-Back corner
            label = depth_map.shape[0] // 2 * label + depth_map.shape[0] // 2
            keypoints = label[:, :2].tolist()

            # keypoints2 = np.asarray(keypoints)
            # vis_keypoints(depth_map, keypoints2[:, :2], show=False)

            transformed = self.transform(image=depth_map, keypoints=keypoints)
            depth_map = transformed["image"]
            keypoints = transformed["keypoints"]

            keypoints = np.asarray(keypoints)
            # vis_keypoints(depth_map, keypoints[:, :2])

            label[:, 0] = keypoints[:, 0]
            label[:, 1] = keypoints[:, 1]

            label = (label - depth_map.shape[0] // 2) / (depth_map.shape[0] // 2)

            # the 3D augmentations have to be done separately,
            # since albumentations can work only with 2D images
            if random.random() < self.p_augment_3d:
                depth_map, label = aug_translate_depth(depth_map, label)

            # label = depth_map.shape[0] // 2 * label + depth_map.shape[0] // 2
            # keypoints = label[:, :2].tolist()
            #
            # keypoints2 = np.asarray(keypoints)
            # vis_keypoints(depth_map, keypoints2[:, :2], show=True)

            # label = (label - depth_map.shape[0] // 2) / (depth_map.shape[0] // 2)

        depth_map = torch.from_numpy(depth_map)
        depth_map = depth_map.unsqueeze(0).expand(3, self.data_resolution, self.data_resolution)

        label = torch.from_numpy(np.asarray(label))

        return depth_map, label

    def __len__(self):
        return len(self.data)


class HPOESSequentialDataset(torch_data.Dataset):
    """Object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties. The data are in format defined by Oberweger: 224x224 depth images (-1, 1)
    with labels -1.0 to +1.0 from cube = (250, 250, 250)"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, sequence_length=3, encoded=True, transform=0.0,
                 p_augment_3d=0.0, mode: str = 'test'):
        """
        Initiates the HPOESSequentialDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param sequence_length: Length of video sequence
        :param encoded: Whether to read only encoded data and decode them at runtime (default: True)
        :param transform: Probability of 2D augmentations (default: 0.0)
        :param p_augment_3d: Probability of 3D augmentations (default: 0.0)
        :param mode: train, eval, test
        """
        self.encoded = encoded

        if not encoded:
            loaded_data = load_hpoes_data(dataset_filename)
        else:
            loaded_data = load_encoded_hpoes_data(dataset_filename)

        data = loaded_data["data"]
        if mode == 'test':
            labels = None
        else:
            labels = loaded_data["labels"]
            # Prevent from initiating with invalid data
            if len(data) != len(labels):
                logging.error("The size of the data (depth maps) list is not equal to the size of the labels list.")

        if (sequence_length % 2) == 0:
            logging.info("Sequence length has to be odd. We will replace it for you.")
            sequence_length = 3

        self.sequence_length = sequence_length
        self.data = data
        self.labels = labels
        self.p_augment_3d = p_augment_3d
        self.transform = transform
        self.mode = mode
        self.rand_choice = 0
        self.dilate_size = 3
        self.erode_size = 3

    def get_indexes(self, idx):
        indexes = np.arange(self.sequence_length) + idx - ((self.sequence_length - 1) / 2)
        indexes[indexes < 0] = 0
        indexes[indexes >= len(self.labels)] = len(self.labels) - 1
        # indexes = np.zeros(self.sequence_length) + idx
        return indexes.astype(int)

    def aug_dilate(self, image, **kwargs):
        image = image.copy()
        image = -image
        image = cv2.dilate(image, np.ones((self.dilate_size, self.dilate_size)))
        image = -image

        return image

    def aug_erode(self, image, **kwargs):
        image = image.copy()
        image = -image
        image = cv2.erode(image, np.ones((self.erode_size, self.erode_size)))
        image = -image

        return image

    @staticmethod
    def aug_morph_close(image, **kwargs):
        dilate_size = 5

        image = image.copy()
        image = -image
        image = cv2.dilate(image, np.ones((dilate_size, dilate_size)))
        image = cv2.erode(image, np.ones((dilate_size, dilate_size)))
        image = -image

        return image

    @staticmethod
    def aug_keypoints(keypoints, **kwargs):
        return keypoints

    def sequence_augmentation(self, p_apply=0.5, limit_rotation=40, limit_translation=0.1, limit_scale=(-0.2, 0.2)):
        if self.rand_choice == 1:
            augm = A.Lambda(image=self.aug_dilate, keypoint=self.aug_keypoints)
        elif self.rand_choice == 2:
            augm = A.Lambda(image=self.aug_erode, keypoint=self.aug_keypoints)
        else:
            augm = A.NoOp()
        transform = A.Compose([
            A.Lambda(image=self.aug_morph_close, keypoint=self.aug_keypoints, p=1.0),
            augm,
            A.Downscale(scale_min=0.5, scale_max=0.9, p=p_apply, interpolation=cv2.INTER_NEAREST_EXACT),
            A.ShiftScaleRotate(limit_translation, limit_scale, limit_rotation, p=p_apply,
                               border_mode=cv2.BORDER_REFLECT101,
                               value=-1.0)
        ], additional_targets={'image1': 'image', 'image2': 'image'},
            keypoint_params=A.KeypointParams("xy", remove_invisible=False))

        return transform

    def sequence_preprocessing(self):
        transform = A.Compose([
            A.Lambda(image=self.aug_morph_close, keypoint=self.aug_keypoints, p=1.0),
        ], additional_targets={'image1': 'image', 'image2': 'image'})
        return transform

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """
        indexes = self.get_indexes(idx)

        if self.encoded:
            _file = io.BytesIO(self.data[indexes[0]])
            depth_map = np.load(_file)["arr_0"]
        else:
            depth_map = self.data[indexes[0]]
        depth_map = np.expand_dims(depth_map, axis=0)

        for index in indexes[1:]:
            if self.encoded:
                _file = io.BytesIO(self.data[index])
                next_frame = np.load(_file)["arr_0"]
                next_frame = np.expand_dims(next_frame, axis=0)
                depth_map = np.vstack((depth_map, next_frame))

            else:
                next_frame = self.data[index]
                next_frame = np.expand_dims(next_frame, axis=0)
                depth_map = np.vstack((depth_map, next_frame))

        if self.labels:
            label = self.labels[idx]
        else:
            label = None

        if self.mode == 'test' or self.mode == 'eval':
            preprocessing = self.sequence_preprocessing()
            transformed = preprocessing(image=depth_map[0], image1=depth_map[1], image2=depth_map[2])
            depth_map[0] = transformed["image"]
            depth_map[1] = transformed["image1"]
            depth_map[2] = transformed["image2"]
            depth_map = torch.from_numpy(depth_map)
            if label is not None:
                label = torch.from_numpy(np.asarray(label))
                return depth_map, label
            return depth_map

        # Perform any additionally desired transformations
        if self.transform > 0:
            self.rand_choice = np.random.choice(3, 1)
            self.dilate_size = np.random.randint(3, 7)
            self.erode_size = np.random.randint(3, 5)
            det_transform = self.sequence_augmentation(p_apply=self.transform)
            # transform the labels into image coordinate
            # the labels are expected to be in relative coordinates of the volume they stem from
            # with (-1, -1, -1) in the Top-Left-Front corner of the volume
            # (0, 0, 0) in the center and (1, 1, 1) in the Bottom-Right-Back corner
            label = depth_map.shape[1] // 2 * label + depth_map.shape[1] // 2
            keypoints = label[:, :2].tolist()

            transformed = det_transform(image=depth_map[0], image1=depth_map[1], image2=depth_map[2],
                                        keypoints=keypoints)
            depth_map[0] = transformed["image"]
            depth_map[1] = transformed["image1"]
            depth_map[2] = transformed["image2"]
            keypoints = transformed["keypoints"]

            keypoints = np.asarray(keypoints)

            label[:, 0] = keypoints[:, 0]
            label[:, 1] = keypoints[:, 1]

            label = (label - depth_map.shape[1] // 2) / (depth_map.shape[1] // 2)

            # the 3D augmentations have to be done separately,
            # since albumentations can work only with 2D images
            if random.random() < self.p_augment_3d:
                depth_map, label = aug_translate_depth(depth_map, label)

        depth_map = torch.from_numpy(depth_map)
        label = torch.from_numpy(np.asarray(label))

        return depth_map, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    pass

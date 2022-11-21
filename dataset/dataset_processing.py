import h5py
import io
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2


def visualize_hpoes_data(filename: str):
    """
    Visualizes the data present in the given h5 file (cutouts of the depth maps, where hands interact with objects).

    :param filename: Path to the h5 file
    """

    data_file = h5py.File(filename)

    for record_index in range(len(data_file["images"])):
        pdata = data_file["images"][str(record_index)][:].tostring()
        _file = io.BytesIO(pdata)
        data = np.load(_file)["arr_0"]

        plt.imshow(data, cmap=matplotlib.cm.jet, interpolation="nearest")
        plt.show()


def load_hpoes_data(filename: str, mode: str):
    """
    Processes the data present in the given h5 file (cutouts of the depth maps, where hands interact with objects).

    :param mode: train, eval, test
    :param filename: Path to the h5 file
    :return: Dictionary with the following items:
        - data: List of numpy.ndarrays with the depth maps (224x224)
        - labels: List of numpy.ndarrays with the 3D coordinates of the individual hand joints (in millimeters, relative
        to the center of the image) (21x3)
    """

    data_file = h5py.File(filename)

    output_depth_maps = []
    output_labels = []
    output_cubes = []

    for record_index in range(len(data_file["images"])):
        pdata = data_file["images"][str(record_index)][:].tostring()
        _file = io.BytesIO(pdata)
        data = np.load(_file)["arr_0"]
        output_depth_maps.append(data)

        output_cubes.append(data_file["cube"][record_index])

        if mode != 'test':
            labels = data_file["labels"][record_index]
            output_labels.append(labels)

    return {"data": output_depth_maps, "labels": output_labels, "cubes": output_cubes}


def load_encoded_hpoes_data(filename: str, mode: str):
    """
    Reads the encoded hand pose data to memory. The decoding needs to be performed outside of this function. This is
    suitable, when the decoded data do not fit into memory.

    :param mode: train, eval, test
    :param filename: Path to the h5 file
    :return: Dictionary with the following items:
        - data: List of bytes with the encoded depth maps (224x224)
        - labels: List of numpy.ndarrays with the 3D coordinates of the individual hand joints (in millimeters, relative
        to the center of the image) (21x3)
    """

    data_file = h5py.File(filename)

    output_depth_maps = []
    output_labels = []
    output_cubes = []

    for record_index in range(len(data_file["images"])):
        pdata = data_file["images"][str(record_index)][:].tostring()
        output_depth_maps.append(pdata)

        output_cubes.append(data_file["cube"][record_index])

        if mode != 'test':
            labels = data_file["labels"][record_index]
            output_labels.append(labels)

    return {"data": output_depth_maps, "labels": output_labels, "cubes": output_cubes}


def aug_dilate(image, **kwargs):
    dilate_size = np.random.randint(3, 7)

    image = image.copy()
    image = -image
    image = cv2.dilate(image, np.ones((dilate_size, dilate_size)))
    image = -image

    return image


def aug_erode(image, **kwargs):
    erode_size = np.random.randint(3, 5)

    image = image.copy()
    image = -image
    image = cv2.erode(image, np.ones((erode_size, erode_size)))
    image = -image

    return image


def aug_erode_or_dilate(image, **kwargs):
    dilate_size = np.random.randint(3, 7)
    erode_size = np.random.randint(3, 5)
    choice = np.random.randint(2)

    image = image.copy()
    image = -image
    if choice == 0:
        image = cv2.dilate(image, np.ones((dilate_size, dilate_size)))
    else:
        image = cv2.erode(image, np.ones((erode_size, erode_size)))

    image = -image

    return image


def aug_morph_close(image, **kwargs):
    dilate_size = 5

    image = image.copy()
    image = -image
    image = cv2.dilate(image, np.ones((dilate_size, dilate_size)))
    image = cv2.erode(image, np.ones((dilate_size, dilate_size)))
    image = -image

    return image


def cropout(image, **kwargs):
    transform = A.CoarseDropout(max_holes=8, min_holes=6, max_height=32, max_width=32, min_height=8, min_width=8,
                                mask_fill_value=[-1, 1])
    image = image.copy()
    transformed = transform(image=image)
    image = transformed["image"]
    return image


def aug_translate_depth(image, keypoints, depth_mean=0.0, depth_std=0.1):
    random_translation = depth_mean + depth_std * np.random.randn()

    image = image.copy()
    mask = np.where((-1.0 < image) & (image < 1.0))
    image[mask] += random_translation
    image[image < -1.0] = -1.0
    image[image > 1.0] = 1.0

    keypoints[:, 2] += random_translation
    keypoints[keypoints[:, 2] < -1.0, 2] = -1.0
    keypoints[keypoints[:, 2] > 1.0, 2] = 1.0

    return image, keypoints


def aug_keypoints(keypoints, **kwargs):
    return keypoints


def augmentation(p_apply=0.5, limit_rotation=40, limit_translation=0.1, limit_scale=(-0.2, 0.2)):
    transform = A.Compose([
        A.Lambda(image=aug_morph_close, keypoint=aug_keypoints, p=1.0),
        A.OneOf([
            A.Lambda(image=aug_dilate, keypoint=aug_keypoints),
            A.Lambda(image=aug_erode, keypoint=aug_keypoints),
            A.NoOp()
        ], p=p_apply),
        # A.Lambda(image=aug_erode_or_dilate, keypoint=aug_keypoints, p=p_apply),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=p_apply, interpolation=cv2.INTER_NEAREST_EXACT),
        A.ShiftScaleRotate(limit_translation, limit_scale, limit_rotation, p=p_apply, border_mode=cv2.BORDER_REFLECT101,
                           value=-1.0),
        A.Lambda(image=cropout, keypoint=aug_keypoints, p=p_apply),
    ], keypoint_params=A.KeypointParams("xy", remove_invisible=False))

    return transform


if __name__ == "__main__":
    pass

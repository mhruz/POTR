
import h5py
import io
import matplotlib

import numpy as np
import matplotlib.pyplot as plt


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


def load_hpoes_data(filename: str):
    """
    Processes the data present in the given h5 file (cutouts of the depth maps, where hands interact with objects).

    :param filename: Path to the h5 file
    :return: Dictionary with the following items:
        - data: List of numpy.ndarrays with the depth maps (224x224)
        - labels: List of numpy.ndarrays with the 3D coordinates of the individual hand joints (in millimeters, relative
        to the center of the image) (21x3)
    """

    data_file = h5py.File(filename)

    output_depth_maps = []
    output_labels = []

    for record_index in range(len(data_file["images"])):
        pdata = data_file["images"][str(record_index)][:].tostring()
        _file = io.BytesIO(pdata)
        data = np.load(_file)["arr_0"]
        labels = data_file["labels"][record_index]

        output_depth_maps.append(data)
        output_labels.append(labels)

    return {"data": output_depth_maps, "labels": output_labels}


if __name__ == "__main__":
    pass


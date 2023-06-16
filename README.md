# POTR
Pose Transformer


## Standard POTR

## Deformable POTR

The implementation is based on [this implementation](https://github.com/fundamentalvision/Deformable-DETR) of Deformable DETR.

### Building

Because the original implementation structures the `MSDeformAttention` module as a separate Python package with its own Python, CUDA, and C++ scripts, one needs to build this before running the training.

First create a dedicated **Conda environment** with Python 3.7:

```bash
conda create -n deformable_potr python=3.7 pip
conda activate deformable_potr
```

Then install all the dependencies using:

```bash
conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
pip install -r requirements.txt
```

And finally, build the `MSDeformAttention` module (and all other custom CUDA operators) using:

```bash
cd ./deformable_potr/models/ops
sh ./make.sh
```

You can then test the CUDA operators by running `python test.py`, where all tests should pass.

### Data Preparation
```bash
cd ./utils
python crops2h5.py ./nyu/train_comrefV2V_3Dproj.json
```

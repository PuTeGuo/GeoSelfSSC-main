# Geo-SelfSSC: Integrating Dense Geometric Priors for Enhanced Self-Supervised Semantic Scene Completion

This is the official implementation for the paper. 

# 🏗️️ Setup

### Python Environment

We use **Conda** to manage our Python environment:
```shell
conda env create -f environment.yml
```
Then, activate the conda environment :
```shell
conda activate GeoSelfSSC
```

### 💾 Datasets

All non-standard data (like precomputed poses and datasplits) comes with this repository and can be found in the `datasets/` folder.
In addition, please adjust the `data_path` ,`data_depth_path`, `data_depth_std_path`, `data_normal_path` and `data_segmentation_path` in `configs/data/kitti_360.yaml`.\
We explain how to obtain these datasets in [KITTI-360](#KITTI-360) ,[Geometric Cues](#Geometric-Cues)
and [Pseudo-Ground-Truth Segmentation masks](#pseudo-ground-truth-segmentation-masks).

For the `data_path` the folder you link to should have the following structure:

```shell
calibration 
data_2d_raw
data_2d_semantics
data_2d_depth
data_2d_depth_std
data_2d_normal
data_3d_bboxes
data_3d_raw
data_poses
```
For the `data_depth_path`, `data_depth_std_path`, `data_normal_path` or `data_segmentation_path` the folder you link to should have the following structure:

```shell
2013_05_28_drive_0000_sync  2013_05_28_drive_0004_sync  2013_05_28_drive_0007_sync
2013_05_28_drive_0002_sync  2013_05_28_drive_0005_sync  2013_05_28_drive_0009_sync
2013_05_28_drive_0003_sync  2013_05_28_drive_0006_sync  2013_05_28_drive_0010_sync
```


### KITTI-360

To download KITTI-360, go to https://www.cvlibs.net/datasets/kitti-360/index.php and create an account.
We require the perspective images, fisheye images, raw velodyne scans, calibrations, and vehicle poses.

### Geometric Cues
We use the existing SOTA depth estimators ([supervised](https://github.com/hisfog/SfMNeXt-Impl) and [self-supervised](https://github.com/nianticlabs/monodepth2)) to predict the depth cues. 
After the depth map is obtained, the normal result is further predicted.


### Pseudo-Ground-Truth Segmentation masks

We use the [Panoptic Deeplab model zoo (CVPR 2020)](https://github.com/bowenc0221/panoptic-deeplab/tree/master).
First create and activate a new conda environment following the instructions laid out [here](https://github.com/bowenc0221/panoptic-deeplab/blob/master/tools/docs/INSTALL.md). \
You can find the `requirements.txt` file under `\datasets\panoptic-deeplab\requirements.txt`.
You also need to download the [R101-os32 cityscapes baseline model](https://github.com/bowenc0221/panoptic-deeplab/blob/master/tools/docs/MODEL_ZOO.md).

Afterwards, you can run:

```bash
python <path-to-script>/preprocess_kitti_360_segmentation.py \
--cfg datasets/panoptic-deeplab/configs/panoptic_deeplab_R101_os32_cityscapes.yaml \
--output-dir <path-to-output-directory> \
--checkpoint <path-to-downloaded-model>/panoptic_deeplab_R101_os32_cityscapes.pth
```

# 🏋 Training

The training configuration for the model reported on in the paper can be found in the `configs` folder.
Generally, all trainings are run on 2 Nvidia RTX4090 Gpus with a total memory of 48GB. 
For faster convergence and slightly better results, we use the pretrained model from [BehindTheScenes](https://fwmb.github.io/bts/)
as a backbone from which we start our training. To download the backbone please run:

```bash
./download_backbone.sh
```

**KITTI-360**

```bash
python train.py -cn exp_kitti_360
```

# 🏃 Running the Example

We provide a script to run our pretrained models with custom data.
The script can be found under `scripts/images/gen_img_custom.py` and takes the following flags:

- `--img <path>` / `i <path>`: Path to input image. The image will be resized to match the model's default resolution.
- `--plot` / `-p`: Plot outputs instead of saving them.
- `--model` / `-m`: Path to the model you want to use.

`media/example/` contains two example images. Note that we use the default projection matrices for the respective datasets 
to compute the density profiles (birds-eye views). 
Therefore, if your custom data comes from a camera with different intrinsics, the output profiles might be skewed.

```bash
# Plot outputs
python scripts/images/gen_img_custom.py --img media/example/0000.png --model /out/kitti_360/kitti_360_backend-None-1_20240903-201557 --plot

# Save outputs to disk
python scripts/images/gen_img_custom.py --img media/example/0000.png --model --model /out/kitti_360/<model-name>
```

# 📊 Evaluation

We provide **not only** a way to evaluate our method (Geo-SelfSSC) on the SSCBench KITTI-360 dataset, 
but also a way to easily evaluate/compare other methods. For this, you only need the predictions on the test set 
(sequence 09) saved as `frame_id.npy` files in a folder. \

## Geo-SelfSSC on SSCBench KITTI-360

To evaluate our model on the SSCBench KITTI-360 dataset, we need additional data:

### SSCBench KITTI-360 dataset

We require the SSCBench KITTI-360 dataset, which can be downloaded from [here](https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360).
The folder structure you will have to link to looks like:

```bash
calibration  data_2d_raw  preprocess
```

### SSCBench KITTI-360 ground truth

We also need preprocessed ground truth (voxelized ground truth) that belongs to the KITTI-360 SSCBench data. 
The preprocessed data for KITTI-360 in the GitHub Repo was incorrectly generated ([see here](https://github.com/ai4ce/SSCBench/issues/9)).\
Therefore, we provide the pre-processed ground truth verified in s4c for download [here](https://cvg.cit.tum.de/webshare/g/s4c/voxel_gt.zip).

The folder structure you will have to link to looks like:

```bash
2013_05_28_drive_0000_sync  2013_05_28_drive_0002_sync  2013_05_28_drive_0004_sync  2013_05_28_drive_0006_sync  2013_05_28_drive_0009_sync
2013_05_28_drive_0001_sync  2013_05_28_drive_0003_sync  2013_05_28_drive_0005_sync  2013_05_28_drive_0007_sync  2013_05_28_drive_0010_sync
```

You can now run the evaluation script found at `scripts/benchmarks/sscbench/evaluate_model_sscbench.py` by running:

```bash
python evaluate_model_sscbench.py \
-ssc <path-to-kitti_360-sscbench-dataset> \
-vgt <path-to-preprocessed-voxel-ground-truth> \
-cp <path-to-model-checkpoint> \
-f
```


# 🗣️ Acknowledgements

This repository is based on the [S4C](https://github.com/ahayler/s4c) and [BehindTheScenes](https://fwmb.github.io/bts/). 
We evaluate our models on the novel [SSCBench KITTI-360 benchmark](https://github.com/ai4ce/SSCBench). 
We generate our pseudo 2D ground truth using the [Monodepth2](https://github.com/nianticlabs/monodepth2), [SQLdepth](https://github.com/hisfog/SfMNeXt-Impl) and [Panoptic Deeplab model zoo](https://github.com/bowenc0221/panoptic-deeplab/tree/master).


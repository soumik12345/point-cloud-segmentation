# Point Cloud Segmentation

_**By [Soumik Rakshit](https://github.com/soumik12345) & [Sayak Paul](https://github.com/sayakpaul)**_

<img src="https://github.com/soumik12345/point-cloud-segmentation/workflows/tests/badge.svg" alt="build-failing">

This repository provides a TF2 implementation of PointNet<sup>1</sup> for segmenting point clouds. Our implementation is fully supported on
TPUs allowing you to train models faster. Distributed training (single-device multi-worker) on GPUs is also supported and so is single-GPU
training. For results and pre-trained models please see below.

To get an understanding of PointNet for segmentation, follow this blog post from keras.io: [Point cloud segmentation with PointNet](https://keras.io/examples/vision/pointnet_segmentation/).

We use the **ShapeNetCore dataset** to train our models on individual categories. The dataset is available [here](https://shapenet.org/). To train and test 
our code, you don't need to download the dataset beforehand, though.

**Update November 16, 2021**: We won the [#TFCommunitySpolight award](https://twitter.com/TensorFlow/status/1460321709488152579) for this project.

## Running using Docker

- Build image using `docker build -t point-cloud-image .`

- Run Jupyter Server using `docker run -it --gpus all -p 8888:8888 -v $(pwd):/usr/src/point-cloud-segmentation point-cloud-image`


## Create TFRecords for ShapeNetCore Shape Segmentation

This part is only required if you would like to train models using TPUs. Be advised that
training using TPUs is usually recommended when you have sufficient amount of data. Therefore, 
you should only use TPUs for the following object categories:

* `Airplane`
* `Car`
* `Chair`
* `Table`

As such we only provide results and models for these categories. 

```
Usage: create_tfrecords.py [OPTIONS]

Example:
  python create_tfrecords.py --experiment_configs configs/shapenetcore.py
```


## Train for ShapeNetCore Shape Segmentation

```
Usage: train_shapenet_core.py [OPTIONS]

Options:
  --experiment_configs    Experiment configs (configs/shapenetcore.py)
  --wandb_project_name    Project Name (DEFAULT: pointnet_shapenet_core)
  --use_wandb             Use WandB flag (DEFAULT: True)

Example:
  python train_shapenet_core.py --experiment_configs configs/shapenetcore.py
```

In case you want to change the configuration-related parameters, either edit them directly in
`configs/shapenetcore.py` or add a new configuration and specify the name of the configuration
in the command line.

## Notes on the Training Setup

* The `batch_size` in the configuration denotes local batch size. If you are using single-host multi-worker distributed training,
the `batch_size` denoted here will be multiplied by the number of workers you have. 
* Using a Google Cloud Storage (GCS) based `artifact_location` is not a requirement if you are using GPU(s). But for 
TPUs, it's a requirement. 

## Notebooks

We also provide notebooks for training and testing the models:

* `notebooks/train_gpu.ipynb` lets you train using GPU(s). If you are using multiple GPUs in the single machine it will
be detected automatically. If your machine supports mixed-precision, then also it will be detected automatically.
* `notebooks/train_tpu.ipynb` lets you train using TPUs. For this using TFRecords for handling data IS a requirement.
* `notebooks/run_inference.ipynb` lets you test the models on GPU(s) on individual object categories.
* `notebooks/keras-tuner.ipynb` lets you tune the hyperparameters of the training routine namely 
  number of epochs, initial learning rate (LR), and LR decaying epochs. We use Keras Tuner for
  this.

We track our training results using [Weights and Biases](https://wandb.ai/) (WandB). For the hyperparameter
tuning part, we combine TensorBoard and WandB.

## Segmentation Results and Models

| <h3>Object Category</h3> | <h3>Training Result</h3> | <h3>Final Model</h3> |
|:---:|:---:|:---:|
| Airplane | [WandB Run](https://wandb.ai/pointnet/pointnet_shapenet_core/runs/n4bm5z0h) | [SavedModel Link](https://github.com/soumik12345/point-cloud-segmentation/releases/download/v0.3/airplane.gz) |
| Car | [WandB Run](https://wandb.ai/pointnet/pointnet_shapenet_core/runs/3vbeyj5w) | [SavedModel Link](https://github.com/soumik12345/point-cloud-segmentation/releases/download/v0.3/car.gz) |
| Chair | [WandB Run](https://wandb.ai/pointnet/pointnet_shapenet_core/runs/1869fpu3) | [SavedModel Link](https://github.com/soumik12345/point-cloud-segmentation/releases/download/v0.3/chair.tar.gz) |
| Table | [WandB Run](https://wandb.ai/pointnet/pointnet_shapenet_core/runs/3sqgxjkb) | [SavedModel Link](https://github.com/soumik12345/point-cloud-segmentation/releases/download/v0.3/table.tar.gz) |

Below are some segmentation results:

### Airplane

![](./assets/Airplane/airplane.gif)

### Car

![](./assets/Car/car.gif)

### Chair

![](./assets/Chair/chair.gif)

### Table

![](./assets/Table/table.gif)

## References

[1] PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation; Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas;
CVPR 2017; https://arxiv.org/abs/1612.00593.

## Acknowledgements

We are thankful to the [GDE program](https://developers.google.com/programs/experts/) for providing us GCP credits.

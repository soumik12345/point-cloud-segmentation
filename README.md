# Point Cloud Segmentation

_**By [Soumik Rakshit](https://github.com/soumik12345) & [Sayak Paul](https://github.com/sayakpaul)**_

<img src="https://github.com/soumik12345/point-cloud-segmentation/workflows/tests/badge.svg" alt="build-failing">

This repository provides a TF2 implementation of PointNet<sup>1</sup> for segmenting point clouds. Our implementation is fully supported on
TPUs allowing you to train models faster. Distributed training (single-device multi-worker) on GPUs is also supported and so is single-GPU
training. 

To get an understanding of PointNet for segmentation, follow this blog post from keras.io: [Point cloud segmentation with PointNet](https://keras.io/examples/vision/pointnet_segmentation/).

## Running using Docker

- Build image using `docker build -t point-cloud-image .`

- Run Jupyter Server using `docker run -it --gpus all -p 8888:8888 -v $(pwd):/usr/src/point-cloud-segmentation point-cloud-image`


## Create TFRecords for ShapenetCore Shape Segmentation

```
Usage: create_tfrecords.py [OPTIONS]

Options:
  --experiment_configs.val_split             Validation Split (DEFAULT: 0.2)
  --experiment_configs.object_category       ShapenetCore object category (DEFAULT: 'Airplane')
  --experiment_configs.artifact_location     TFRecord dump dir (DEFAULT: 'gs://pointnet-segmentation/tfrecords')
  --experiment_configs.metadata_url          Metadata URL
  --experiment_configs.samples_per_shard     Max number of data samples per TFRecord file (DEFAULT: 512)

Example:
  python create_tfrecords.py --experiment_configs configs/shapenetcore.py
```


## Train for ShapenetCore Shape Segmentation

```
Usage: train_shapenet_core.py [OPTIONS]

Options:
  --experiment_configs                       Experiment configs (configs/shapenetcore.py)
  --experiment_configs.wandb_project_name    Project Name (DEFAULT: pointnet_shapenet_core)
  --experiment_configs.experiment_name       Experiment Name (DEFAULT: shapenet_core_experiment)
  --experiment_configs.wandb_api_key         W&B API Key (OPTIONAL)
  --experiment_configs.object_category       ShapenetCore object category (DEFAULT: 'Airplane')
  --experiment_configs.in_memory             Flag: Use In-memory dataloader (DEFAULT: True)
  --experiment_configs.batch_size            Batch Size (DEFAULT: 32)
  --experiment_configs.num_points            Number of points to be sampled from a given point cloud (DEFAULT: 1024)
  --experiment_configs.initial_lr            Initial Learning Rate (DEFAULT: 1e-3)
  --experiment_configs.drop_every            Epochs after which Learning Rate is dropped (DEFAULT: 20)
  --experiment_configs.decay_factor          Learning Rate Decay Factor (DEFAULT: 0.5)
  --experiment_configs.epochs                Number of training epochs (DEFAULT: 100)
  --experiment_configs.use_mp                Flag: Use mixed-precision or not (DEFAULT: False)
  --experiment_configs.use_tpus              Flag: Use mixed-precision or not (DEFAULT: True)

Example:
  python train_shapenet_core.py --experiment_configs configs/shapenetcore.py
```

## Notes

* The `batch_size` here denotes local batch size. If you are using single-host multi-worker distributed training,
the `batch_size` denoted here will be multiplied by the number of workers you have. 
* Using a Google Cloud Storage (GCS) based `artifact_location` is not a requirement if you are using GPU(s). But for 
TPUs, it's a requirement. 

## References

[1] PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation; Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas;
CVPR 2017; https://arxiv.org/abs/1612.00593.

## Acknowledgements

We are thankful to the [GDE program](https://developers.google.com/programs/experts/) for providing us GCP credits.

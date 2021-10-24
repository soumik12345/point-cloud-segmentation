# Point Cloud Segmentation

<img src="https://github.com/soumik12345/point-cloud-segmentation/workflows/tests/badge.svg" alt="build-failing">

## Running using Docker

- Build image using `docker build -t point-cloud-image .`

- Run Jupyter Server using `docker run -it --gpus all -p 8888:8888 -v $(pwd):/usr/src/point-cloud-segmentation point-cloud-image`


## Create TFRecords for ShapenetCore Shape Segmentation

```
Usage: train_shapenet_core.py [OPTIONS]

Options:
  --tfrecord_dir          TFRecord dump dir (DEFAULT: './tfrecords')
  --samples_per_shard     Max number of data samples per TFRecord file (DEFAULT: 512)

Example:
  python create_tfrecords.py --experiment_configs configs/shapenetcore.py
```


## Train for ShapenetCore Shape Segmentation

```
Usage: train_shapenet_core.py [OPTIONS]

Options:
  --wandb_project_name    Project Name (DEFAULT: pointnet_shapenet_core)
  --experiment_name       Experiment Name (DEFAULT: shapenet_core_experiment)
  --wandb_api_key         W&B API Key (OPTIONAL)
  --experiment_configs    Experiment configs (configs/shapenetcore.py)
  --object_category       ShapenetCore object category (DEFAULT: 'Airplane')
  --in_memory             Flag: Use In-memory dataloader (DEFAULT: True)
  --batch_size            Batch Size (DEFAULT: 32)
  --num_points            Number of points to be sampled from a given point cloud (DEFAULT: 1024)
  --initial_lr FLOAT      Initial Learning Rate (DEFAULT: 1e-3)
  --drop_every            Epochs after which Learning Rate is dropped (DEFAULT: 20)
  --decay_factor          Learning Rate Decay Factor (DEFAULT: 0.5)
  --epochs                Number of training epochs (DEFAULT: 50)
  --use_mp                Flag: Use mixed-precision or not (DEFAULT: True)

Example:
  python train_shapenet_core.py --experiment_configs configs/shapenetcore.py
```

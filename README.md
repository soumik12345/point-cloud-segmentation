# Point Cloud Segmentation

<img src="https://github.com/soumik12345/point-cloud-segmentation/workflows/tests/badge.svg" alt="build-failing">

## Running using Docker

- Build image using `docker build -t point-cloud-image .`

- Run Jupyter Server using `docker run -it --gpus all -p 8888:8888 -v $(pwd):/usr/src/point-cloud-segmentation point-cloud-image`


## Train for ShapenetCore Shape Segmentation

```
Usage: train_shapenet_core.py [OPTIONS]

Options:
  --experiment_configs        Experiment configs (configs/shapenetcore.py)
  --object_category           ShapenetCore object category
  --in_memory                 Flag: Use In-memory dataloader
  --batch_size                Batch Size
  --num_points                Number of points to be sampled from a given point cloud
  --initial_lr FLOAT          Initial Learning Rate
  --drop_every                Epochs after which Learning Rate is dropped
  --decay_factor              Learning Rate Decay Factor
  --epochs                    Number of training epochs
  --use_baseline_model        Flag: Use Baseline Model or ShapenetCore Segmenbtation Model
```

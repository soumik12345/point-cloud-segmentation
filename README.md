# Point Cloud Segmentation

<img src="https://github.com/soumik12345/point-cloud-segmentation/workflows/tests/badge.svg" alt="build-failing">

## Running using Docker

- Build image using `docker build -t point-cloud-image .`

- Run Jupyter Server using `docker run -it --gpus all -p 8888:8888 -v $(pwd):/usr/src/point-cloud-segmentation point-cloud-image`


## Train PointNet on ShapenetCore Shape Segmentation

```
Usage: train_shapenet_core.py [OPTIONS]

Options:
  -c, --category TEXT             Shapenet Category
  -i, --in_memory                 Flag: Use In-memory dataloader
  -b, --batch_size INTEGER        Batch Size
  -n, --n_sampled_points INTEGER  Number of points to be sampled from a give
                                  point cloud
  -l, --initial_lr FLOAT          Initial Learning Rate
  -d, --drop_every INTEGER        Epochs after which Learning Rate is dropped
  -f, --decay_factor FLOAT        Learning Rate Decay Factor
  -e, --epochs INTEGER            Number of training epochs
  -m, --use_baseline_model        Flag: Use Baseline Model or ShapenetCore
                                  Segmenbtation Model
  --help                          Show this message and exit.
```

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.object_category = "Airplane"  # ShapeNet Category
    config.in_memory = True  # Flag: Use In-memory dataloader
    config.batch_size = 32  # Batch Size
    config.num_points = 1024  # Number of points to be sampled from a given point cloud
    config.val_split = 0.2  # Fraction representing Validation Split

    config.initial_lr = 1e-3  # Initial Learning Rate
    config.drop_every = 20  # Epochs after which Learning Rate is dropped
    config.decay_factor = 0.5  # Learning Rate Decay Factor
    config.epochs = 100  # Number of training epochs
    config.use_mp = False  # Flag: Use mixed-precision or not
    config.use_tpus = True  # Flag: Use TPUs or not

    config.metadata_url = "https://github.com/soumik12345/point-cloud-segmentation/releases/download/v0.2/metadata.json"  # Metadata URL
    config.artifact_location = "gs://pointnet-segmentation"  # Artifact dump dir (using a GCS location is not a requirement
    # GPUs but for running on TPUs data must be coming from a GCS location).
    config.samples_per_shard = 512  # Max number of data shards per TFRecord file
    config.jitter_minval = -5e-3  # Point cloud jitter range lower limit
    config.jitter_maxval = 5e-3  # Point cloud jitter range upper limit

    return config

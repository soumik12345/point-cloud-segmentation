import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.object_category = "Airplane"  # ShapeNet Category
    config.in_memory = True  # Flag: Use In-memory dataloader
    config.batch_size = 32  # Batch Size
    config.num_points = 1024  # Number of points to be sampled from a given point cloud

    config.initial_lr = 1e-3  # Initial Learning Rate
    config.drop_every = 20  # Epochs after which Learning Rate is dropped
    config.decay_factor = 0.5  # Learning Rate Decay Factor
    config.epochs = 50  # Number of training epochs
    config.use_mp = True  # Flag: Use mixed-precision or not
    return config

r"""Train a Pointnet-based Shape Segmentation Model.

Sample Useage:
python train_shapenet_core.py --experiment_configs configs/shapenetcore.py
"""

import os
from absl import app
from absl import flags
from datetime import datetime

from ml_collections.config_flags import config_flags

from tensorflow.keras import optimizers, callbacks

from point_seg import ShapeNetCoreLoaderInMemory, ShapeNetCoreLoader
from point_seg import models, utils


FLAGS = flags.FLAGS
flags.DEFINE_string("wandb_project_name", "pointnet_shapenet_core", "W&B Project Name")
flags.DEFINE_string("experiment_name", "shapenet_core_experiment", "Experiment Name")
flags.DEFINE_string("wandb_api_key", None, "Wandb API Key")
config_flags.DEFINE_config_file("experiment_configs")


def main(_):

    # Initialize W&B
    if FLAGS.wandb_api_key is not None:
        utils.init_wandb(
            FLAGS.wandb_project_name,
            FLAGS.experiment_name,
            FLAGS.wandb_api_key,
            FLAGS.experiment_configs.to_dict(),
        )

    # Define Dataloader
    data_loader = (
        ShapeNetCoreLoaderInMemory(
            object_category=FLAGS.experiment_configs.object_category,
            n_sampled_points=FLAGS.experiment_configs.num_points,
        )
        if FLAGS.experiment_configs.in_memory
        else ShapeNetCoreLoader(
            object_category=FLAGS.experiment_configs.object_category,
            n_sampled_points=FLAGS.experiment_configs.num_points,
        )
    )
    if FLAGS.experiment_configs.in_memory:
        data_loader.load_data()

    # Create tf.data.Datasets
    train_dataset, val_dataset = data_loader.get_datasets(
        batch_size=FLAGS.experiment_configs.batch_size
    )

    # Learning Rate scheduling callback
    lr_scheduler = utils.StepDecay(
        FLAGS.experiment_configs.initial_lr,
        FLAGS.experiment_configs.drop_every,
        FLAGS.experiment_configs.decay_factor,
    )
    lr_callback = callbacks.LearningRateScheduler(
        lambda epoch: lr_scheduler(epoch), verbose=True
    )

    # Tensorboard Callback
    logs_dir = f'logs_{datetime.utcnow().strftime("%y%m%d-%H%M%S")}'
    tb_callback = callbacks.TensorBoard(log_dir=logs_dir)

    # Model Checkpoint Callback
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=os.path.join(logs_dir, "checkpoints", "model_{epoch}"),
        save_weights_only=True,
    )

    # Define Model and Optimizer
    optimizer = optimizers.Adam(learning_rate=FLAGS.experiment_configs.initial_lr)
    _, y = next(iter(train_dataset))
    num_classes = y.shape[-1]
    model = (
        models.get_baseline_segmentation_model(
            FLAGS.experiment_configs.num_points, num_classes
        )
        if FLAGS.experiment_configs.use_baseline_model
        else models.get_shape_segmentation_model(
            FLAGS.experiment_configs.num_points, num_classes
        )
    )
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FLAGS.experiment_configs.epochs,
        callbacks=[tb_callback, lr_callback, checkpoint_callback],
    )


if __name__ == "__main__":
    app.run(main)

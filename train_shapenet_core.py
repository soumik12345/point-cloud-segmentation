r"""Train a Pointnet-based Shape Segmentation Model.

Sample Useage:
python train_shapenet_core.py --experiment_configs configs/shapenetcore.py
"""

import os
from absl import app
from absl import flags
from absl import logging
from datetime import datetime

from ml_collections.config_flags import config_flags

from tensorflow.keras import optimizers, callbacks
from tensorflow.keras import mixed_precision

from point_seg import TFRecordLoader
from point_seg import models, utils


FLAGS = flags.FLAGS
flags.DEFINE_string("wandb_project_name", "pointnet_shapenet_core", "W&B Project Name")
flags.DEFINE_string("experiment_name", "shapenet_core_experiment", "Experiment Name")
flags.DEFINE_string("wandb_api_key", None, "WandB API Key")
config_flags.DEFINE_config_file("experiment_configs")


def main(_):

    strategy = utils.initialize_device()

    # Initialize W&B
    if FLAGS.wandb_api_key is not None:
        utils.init_wandb(
            FLAGS.wandb_project_name,
            FLAGS.experiment_name,
            FLAGS.wandb_api_key,
            FLAGS.experiment_configs.to_dict(),
        )

    if FLAGS.experiment_configs.use_mp and not FLAGS.experiment_configs.use_tpus:
        mixed_precision.set_global_policy("mixed_float16")
        logging.info("Using mixed-precision.")
        policy = mixed_precision.global_policy()
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"
    else:
        raise ValueError(
            "TPUs run with mixed-precision by default. No need to specify precision separately."
        )

    # Define Dataloader
    logging.info("Preparing data loader.")
    tfrecord_loader = TFRecordLoader(
        tfrecord_dir=os.path.join(
            FLAGS.experiment_configs.artifact_location, "tfrecords"
        ),
        object_category=FLAGS.experiment_configs.object_category,
    )
    train_dataset, val_dataset = tfrecord_loader.get_datasets(
        batch_size=FLAGS.experiment_configs.batch_size * strategy.num_replicas_in_sync
    )

    # Learning Rate scheduling callback
    logging.info("Initializing callbacks.")
    lr_scheduler = utils.StepDecay(
        FLAGS.experiment_configs.initial_lr * strategy.num_replicas_in_sync,
        FLAGS.experiment_configs.drop_every,
        FLAGS.experiment_configs.decay_factor,
    )
    lr_callback = callbacks.LearningRateScheduler(
        lambda epoch: lr_scheduler(epoch), verbose=True
    )

    # Tensorboard Callback
    timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")
    logs_dir = f"logs_{timestamp}"
    logs_dir = os.path.join(FLAGS.experiment_configs.artifact_location, logs_dir)
    tb_callback = callbacks.TensorBoard(log_dir=logs_dir)

    # Model Checkpoint Callback
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=os.path.join(
            FLAGS.experiment_configs.artifact_location,
            f"checkpoints_{timestamp}",
            "model_{epoch}",
        ),
        save_weights_only=True,
    )

    # Define Model and Optimizer
    with strategy.scope():
        logging.info("Initializing segmentation model.")
        optimizer = optimizers.Adam(
            learning_rate=FLAGS.experiment_configs.initial_lr
            * strategy.num_replicas_in_sync
        )
        _, y = next(iter(train_dataset))
        num_classes = y.shape[-1]
        model = models.get_shape_segmentation_model(
            FLAGS.experiment_configs.num_points, num_classes
        )

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train
    logging.info("Beginning training.")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FLAGS.experiment_configs.epochs,
        callbacks=[tb_callback, lr_callback, checkpoint_callback],
    )
    logging.info("Training complete.")


if __name__ == "__main__":
    app.run(main)

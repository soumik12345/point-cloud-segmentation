r"""Train a Pointnet-based Shape Segmentation Model.

Sample Usage:
python train_shapenet_core.py --experiment_configs configs/shapenetcore.py
"""

import os

import wandb.keras
from absl import app
from absl import flags
from absl import logging
from datetime import datetime

from ml_collections.config_flags import config_flags

from tensorflow.keras import optimizers, callbacks
from tensorflow.keras import mixed_precision

from point_seg import TFRecordLoader, ShapeNetCoreLoaderInMemory
from point_seg import models, utils


FLAGS = flags.FLAGS
flags.DEFINE_string("wandb_project_name", "pointnet_shapenet_core", "W&B Project Name")
flags.DEFINE_string("wandb_api_key", None, "WandB API Key")
config_flags.DEFINE_config_file("experiment_configs")


def main(_):

    strategy = utils.initialize_device()
    batch_size = FLAGS.experiment_configs.batch_size * strategy.num_replicas_in_sync
    FLAGS.experiment_configs["batch_size"] = batch_size

    # Initialize W&B
    if FLAGS.wandb_api_key is not None:
        utils.init_wandb(
            FLAGS.wandb_project_name,
            FLAGS.experiment_configs.object_category,
            FLAGS.wandb_api_key,
            FLAGS.experiment_configs.to_dict(),
        )

    if FLAGS.experiment_configs.use_mp and not FLAGS.experiment_configs.use_tpus:
        mixed_precision.set_global_policy("mixed_float16")
        logging.info("Using mixed-precision.")
        policy = mixed_precision.global_policy()
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"
    elif FLAGS.experiment_configs.use_mp and FLAGS.experiment_configs.use_tpus:
        raise ValueError(
            "TPUs run with mixed-precision by default. No need to specify precision separately."
        )

    # Define Dataloader
    logging.info(
        f"Object category received: {FLAGS.experiment_configs.object_category}."
    )
    logging.info(f"Preparing data loader with a batch size of {batch_size}.")
    train_dataset, val_dataset = None, None
    if FLAGS.experiment_configs.in_memory:
        data_loader = ShapeNetCoreLoaderInMemory(
            object_category=FLAGS.experiment_configs.object_category,
            n_sampled_points=FLAGS.experiment_configs.num_points,
        )
        data_loader.load_data()
        train_dataset, val_dataset = data_loader.get_datasets(
            val_split=FLAGS.experiment_configs.val_split,
            batch_size=FLAGS.experiment_configs.batch_size,
        )
    else:
        tfrecord_loader = TFRecordLoader(
            tfrecord_dir=os.path.join(
                FLAGS.experiment_configs.artifact_location, "tfrecords"
            ),
            object_category=FLAGS.experiment_configs.object_category,
        )
        drop_remainder = True if FLAGS.experiment_configs.use_tpus else False
        train_dataset, val_dataset = tfrecord_loader.get_datasets(
            batch_size=batch_size, drop_remainder=drop_remainder
        )

    # Learning Rate scheduling callback
    logging.info("Initializing callbacks.")
    lr_scheduler = utils.StepDecay(
        FLAGS.experiment_configs.initial_lr,
        FLAGS.experiment_configs.drop_every,
        FLAGS.experiment_configs.decay_factor,
    )
    lr_callback = callbacks.LearningRateScheduler(
        lambda epoch: lr_scheduler(epoch), verbose=True
    )

    # Tensorboard Callback
    timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")
    logs_dir = f"logs_{FLAGS.experiment_configs.object_category}_{timestamp}"
    logs_dir = os.path.join(FLAGS.experiment_configs.artifact_location, logs_dir)
    tb_callback = callbacks.TensorBoard(log_dir=logs_dir)

    # Model Checkpoint Callback
    checkpoint_path = os.path.join(
        FLAGS.experiment_configs.artifact_location,
        "checkpoints",
        f"{FLAGS.experiment_configs.object_category}_{timestamp}",
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_best_only=True, save_weights_only=True,
    )

    # Pack the callbacks as a list.
    callback_list = [tb_callback, checkpoint_callback, lr_callback]
    if FLAGS.wandb_api_key is not None:
        callback_list.append(wandb.keras.WandbCallback())

    # Define Model and Optimizer
    with strategy.scope():
        logging.info("Initializing segmentation model.")
        optimizer = optimizers.Adam(learning_rate=FLAGS.experiment_configs.initial_lr)
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
        callbacks=callback_list,
    )
    logging.info("Training complete, serializing model with the best checkpoint.")
    serialization_path = os.path.join(
        FLAGS.experiment_configs.artifact_location,
        f"final_model_{FLAGS.experiment_configs.object_category}_{timestamp}",
    )

    # Since the model contains a custom regularizer, during loading the model we need to do the following:
    # model = keras.models.load_model(filepath,
    #   custom_objects={"OrthogonalRegularizer": transform_block.OrthogonalRegularizer}
    # )
    model.load_weights(checkpoint_path)
    model.save(serialization_path)
    logging.info(f"Model serialized to {serialization_path}.")


if __name__ == "__main__":
    app.run(main)

import os
import click
from datetime import datetime

from tensorflow.keras import optimizers, callbacks

from point_seg import ShapeNetCoreLoaderInMemory, ShapeNetCoreLoader
from point_seg import models, utils


@click.command()
@click.option("--category", "-c", default="Airplane", help="Shapenet Category")
@click.option("--in_memory", "-i", is_flag=True, help="Flag: Use In-memory dataloader")
@click.option("--batch_size", "-b", default=16, help="Batch Size")
@click.option(
    "--n_sampled_points",
    "-n",
    default=2048,
    help="Number of points to be sampled from a give point cloud",
)
@click.option("--initial_lr", "-l", default=1e-3, help="Initial Learning Rate")
@click.option(
    "--drop_every", "-d", default=20, help="Epochs after which Learning Rate is dropped"
)
@click.option("--decay_factor", "-f", default=0.5, help="Learning Rate Decay Factor")
@click.option("--epochs", "-e", default=60, help="Number of training epochs")
@click.option(
    "--use_baseline_model",
    "-m",
    is_flag=True,
    help="Flag: Use Baseline Model or ShapenetCore Segmenbtation Model",
)
def train(
    category,
    in_memory,
    batch_size,
    n_sampled_points,
    initial_lr,
    drop_every,
    decay_factor,
    epochs,
    use_baseline_model,
):
    data_loader = (
        ShapeNetCoreLoaderInMemory(
            object_category=category, n_sampled_points=n_sampled_points
        )
        if in_memory
        else ShapeNetCoreLoader(
            object_category=category, n_sampled_points=n_sampled_points
        )
    )
    if in_memory:
        data_loader.load_data()
    train_dataset, val_dataset = data_loader.get_datasets(batch_size=batch_size)
    _, y = next(iter(train_dataset))
    num_classes = y.shape[-1]
    lr_scheduler = utils.StepDecay(initial_lr, drop_every, decay_factor)
    lr_callback = callbacks.LearningRateScheduler(
        lambda epoch: lr_scheduler(epoch), verbose=True
    )
    logs_dir = f'logs_{datetime.utcnow().strftime("%y%m%d-%H%M%S")}'
    tb_callback = callbacks.TensorBoard(log_dir=logs_dir)
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=os.path.join(logs_dir, "checkpoints", "model_{epoch}"),
        save_weights_only=True,
    )
    optimizer = optimizers.Adam(learning_rate=initial_lr)
    model = (
        models.get_baseline_segmentation_model(n_sampled_points, num_classes)
        if use_baseline_model
        else models.get_shape_segmentation_model(n_sampled_points, num_classes)
    )
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[tb_callback, lr_callback, checkpoint_callback],
    )


if __name__ == "__main__":
    train()

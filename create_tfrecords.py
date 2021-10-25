r"""Create TFRecords for a given category of point clouds in the ShapeNetCore dataset.

Sample Useage:
python create_tfrecords.py --experiment_configs configs/shapenetcore.py
"""

from absl import app
from absl import flags

from ml_collections.config_flags import config_flags
import os

from point_seg.dataloader import ShapeNetCoreTFRecordWriter


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")


def main(_):
    tfrecord_writer = ShapeNetCoreTFRecordWriter(
        object_category=FLAGS.experiment_configs.object_category,
        n_sampled_points=FLAGS.experiment_configs.num_points,
    )
    tfrecord_writer.load_data()
    tfrecord_writer.write_tfrecords(
        samples_per_shard=FLAGS.experiment_configs.samples_per_shard,
        tfrecord_dir=os.path.join(
            FLAGS.experiment_configs.artifact_location, "tfrecords"
        ),
        val_split=FLAGS.experiment_configs.val_split,
    )


if __name__ == "__main__":
    app.run(main)

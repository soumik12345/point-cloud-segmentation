import os
import unittest
import tempfile
from glob import glob
import tensorflow as tf

from point_seg import (
    ShapeNetCoreLoaderInMemory,
    ShapeNetCoreLoader,
    ShapeNetCoreTFRecordWriter,
    TFRecordLoader,
)
from point_seg import models


DATASET_URL = "https://github.com/soumik12345/point-cloud-segmentation/releases/download/v0.1/shapenet.zip"


class DataLoaderTester(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        tf.keras.utils.get_file(
            fname="shapenet.zip",
            origin=DATASET_URL,
            cache_subdir="datasets",
            hash_algorithm="auto",
            extract=True,
            archive_format="auto",
            cache_dir="datasets",
        )
        self.in_memory_data_loader = ShapeNetCoreLoaderInMemory(
            object_category="Airplane", viz_samples=100
        )
        self.in_memory_data_loader.load_data()
        self.e2e_data_loader = ShapeNetCoreLoader(object_category="Airplane")

    def test_in_memory_data_loader(self):
        train_dataset, val_dataset = self.in_memory_data_loader.get_datasets()
        x, y = next(iter(train_dataset))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 5)
        x, y = next(iter(val_dataset))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 5)

    def test_e2e_data_loader(self):
        train_dataset, val_dataset = self.e2e_data_loader.get_datasets()
        x, y = next(iter(train_dataset))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 5)
        x, y = next(iter(val_dataset))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 5)


class ShapeSegmentModelTester(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shapenet_model = models.get_shape_segmentation_model(2048, 5)

    def test_model_output_shape(self):
        random_inputs = tf.random.normal((16, 2048, 3))
        random_predictions = self.shapenet_model.predict(random_inputs)
        assert random_predictions.shape == (16, 2048, 5)


class TFRecordTester(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.object_category = "Airplane"
        self.num_points = 1024
        self.samples_per_shard = 512
        self.tfrecord_dir = "/tmp/tfrecords"
        self.val_split = 0.2
        self.batch_size = 16

    def test_tfrecord_creation(self):
        tfrecord_writer = ShapeNetCoreTFRecordWriter(
            object_category=self.object_category, n_sampled_points=self.num_points,
        )
        tfrecord_writer.load_data(limit=100)
        tfrecord_writer.write_tfrecords(
            samples_per_shard=self.samples_per_shard,
            tfrecord_dir=self.tfrecord_dir,
            val_split=self.val_split,
        )
        train_tfrecord_files = glob(
            os.path.join(self.tfrecord_dir, self.object_category, "train/*.tfrec")
        )
        val_tfrecord_files = glob(
            os.path.join(self.tfrecord_dir, self.object_category, "val/*.tfrec")
        )
        assert len(train_tfrecord_files) == 1
        assert len(val_tfrecord_files) == 1

    def test_tfrecord_loader(self):
        loader = TFRecordLoader(self.tfrecord_dir, self.object_category)
        train_ds, val_ds = loader.get_datasets(batch_size=self.batch_size)
        x, y = next(iter(train_ds))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 5)
        x, y = next(iter(val_ds))
        assert x.shape == (16, 1024, 3)
        assert y.shape == (16, 1024, 5)

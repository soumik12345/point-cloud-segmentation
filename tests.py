import tensorflow as tf
import unittest

from point_seg import ShapeNetCoreLoaderInMemory, ShapeNetCoreLoader
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
            object_category="Airplane"
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


class BaselineSegmentModelTester(unittest.TestCase):
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
        self.baseline_model = models.get_baseline_segmentation_model(1024, 5)

    def test_model_output_shape(self):
        random_inputs = tf.random.normal((16, 1024, 3))
        random_predictions = self.baseline_model.predict(random_inputs)
        assert random_predictions.shape == (16, 1024, 5)

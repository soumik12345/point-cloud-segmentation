import os
import json
import random
import numpy as np

import tensorflow as tf

from typing import List
from tqdm import tqdm
from glob import glob

from .augmentations import apply_jitter

_AUTO = tf.data.AUTOTUNE


class ShapeNetCoreLoader:

    """
    End-to-end Dataloader class for ShapeNet Core Dataset.

    Args:
        object_category (str): One of the 12 objects from the ShapenetCore dataset.
        n_sampled_points (int): Number of points to be sampled from each point cloud.
    """

    def __init__(
        self, object_category: str = "Airplane", n_sampled_points: int = 1024
    ) -> None:
        self._get_files()
        self.dataset_path = "/tmp/.keras/datasets/PartAnnotation"
        self.metadata = self._load_metadata()
        if object_category not in self.metadata.keys():
            raise KeyError(
                "Not a valid ShapeNet Object. Must be one of "
                + str(self.metadata.keys())
            )
        else:
            self.object_category = object_category
        self.n_sampled_points = n_sampled_points
        self.point_clouds, self.test_point_clouds = [], []
        self.point_cloud_labels, self.all_labels = [], []
        self.point_cloud_dataframes = []
        self.labels = self.metadata[self.object_category]["lables"]
        self.colors = self.metadata[self.object_category]["colors"]
        self.points_dir = os.path.join(
            self.dataset_path,
            "{}/points".format(self.metadata[object_category]["directory"]),
        )
        self.labels_dir = os.path.join(
            self.dataset_path,
            "{}/points_label".format(self.metadata[object_category]["directory"]),
        )
        self.points_files_with_keys, self.points_files_without_keys = set(), set()

    def _get_files(self):
        dataset_url = "https://github.com/soumik12345/point-cloud-segmentation/releases/download/v0.1/shapenet.zip"
        tf.keras.utils.get_file(
            fname="shapenet.zip",
            origin=dataset_url,
            cache_subdir="datasets",
            hash_algorithm="auto",
            extract=True,
            archive_format="auto",
            cache_dir="datasets",
        )

    def _load_metadata(self):
        with open(os.path.join(self.dataset_path, "metadata.json")) as json_file:
            metadata = json.load(json_file)
        return metadata

    def _load_point_files(self):
        points_files = glob(os.path.join(self.points_dir, "*.pts"))
        for point_file in tqdm(points_files):
            file_id = point_file.split("/")[-1].split(".")[0]
            label_data = {}
            for label in self.labels:
                label_file = os.path.join(self.labels_dir, label, file_id + ".seg")
                if os.path.exists(label_file):
                    label_data[
                        label
                    ] = 0  # Dummy assignment only used as a placeholder.
            try:
                _ = np.vstack(tuple([label_data[key] for key in self.labels]))
                self.points_files_with_keys.add(point_file)
            except KeyError:
                self.points_files_without_keys.add(point_file)
        self.points_files_with_keys = list(self.points_files_with_keys)
        self.points_files_without_keys = list(self.points_files_without_keys)
        print("\nNumber of Point Files with keys:", len(self.points_files_with_keys))
        print(
            "Number of Point Files without keys:", len(self.points_files_without_keys)
        )

    def _random_sampler(self, point_cloud: np.ndarray, label_cloud: np.ndarray):
        n_points = len(point_cloud)
        # Randomly sampling respective indices.
        sampled_indices = random.sample(list(range(n_points)), self.n_sampled_points)
        # Sampling points corresponding to sampled indices.
        sampled_point_cloud = np.array([point_cloud[i] for i in sampled_indices])
        # Sampling corresponding one-hot encoded labels.
        sampled_label_cloud = np.array([label_cloud[i] for i in sampled_indices])
        return sampled_point_cloud, sampled_label_cloud

    def _process_single_point_file(self, point_filepath: str):
        # Load the point cloud from disk.
        point_filepath = point_filepath.numpy().decode("utf-8")
        point_cloud = np.loadtxt(point_filepath)
        # Parse the file-id.
        file_id = point_filepath.split("/")[-1].split(".")[0]
        label_data, num_labels = {}, 0
        # Parse the labels.
        for label in self.labels:
            label_file = os.path.join(self.labels_dir, label, file_id + ".seg")
            label_data[label] = np.loadtxt(label_file).astype("float32")
            num_labels = len(label_data[label])
        label_map = ["none"] * num_labels
        for label in self.labels:
            for i, data in enumerate(label_data[label]):
                label_map[i] = label if data == 1 else label_map[i]
        label_data = np.vstack(tuple([label_data[key] for key in self.labels]))
        label_cloud = label_data.reshape(label_data.shape[1], label_data.shape[0])
        # Sample `N_SAMPLE_POINTS` from the point and label clouds randomly.
        sampled_point_cloud, sampled_label_cloud = self._random_sampler(
            point_cloud, label_cloud
        )
        # Normalizing point cloud.
        normalized_point_cloud = sampled_point_cloud - np.mean(
            sampled_point_cloud, axis=0
        )
        normalized_point_cloud /= np.max(np.linalg.norm(normalized_point_cloud, axis=1))
        return normalized_point_cloud, sampled_label_cloud

    def _tf_process_point_file(self, point_filepath: str):
        return tf.py_function(
            self._process_single_point_file, [point_filepath], [tf.float64, tf.float32]
        )

    def _prepare_dataset(
        self, point_filepaths: List[str], is_train: bool, batch_size: int
    ):
        point_files_ds = tf.data.Dataset.from_tensor_slices(point_filepaths)
        if is_train:
            point_files_ds = point_files_ds.shuffle(batch_size * 100)

        point_ds = point_files_ds.map(
            self._tf_process_point_file, num_parallel_calls=_AUTO
        )
        point_ds = point_ds.batch(batch_size)
        if is_train:
            point_ds = point_ds.map(apply_jitter, num_parallel_calls=_AUTO)
        return point_ds

    def get_datasets(self, val_split: float = 0.2, batch_size: int = 16):
        """
        Get TensorFlow BatchDataset objects for train and validation data.

        Args:
            val_split (str): Fraction representing validation split (default=0.2).
            batch_size (int): Batch size for training and validation (default=16).
        
        Returns:
            train_dataset (TensorFlow BatchDataset): Train dataset,
            val_dataset (TensorFlow BatchDataset): Validation dataset
        """
        self._load_point_files()
        split_index = int(len(self.points_files_with_keys) * (1 - val_split))
        train_point_cloud_files = self.points_files_with_keys[:split_index]
        val_point_cloud_files = self.points_files_with_keys[split_index:]

        print(f"Total training files: {len(train_point_cloud_files)}.")
        print(f"Total validation files: {len(val_point_cloud_files)}.")

        train_dataset = self._prepare_dataset(
            train_point_cloud_files, is_train=True, batch_size=batch_size
        )
        val_dataset = self._prepare_dataset(
            val_point_cloud_files, is_train=False, batch_size=batch_size
        )
        return train_dataset, val_dataset

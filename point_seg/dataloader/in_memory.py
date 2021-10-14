import os
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt


class ShapeNetCoreLoaderInMemory:
    """
    In-memory Dataloader class for Shapenet Core Dataset. This dataloader 
    was built primarily for EDA purposes and hence has good support for 
    visualization. This dataloader loads all the point clouds and corresponding 
    label clouds as in-memory data structures which are subsequently used to 
    generate the TensorFlow BatchDatasets.

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
                "Not a valid Shapenet Object. Must be one of "
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

    def _sample_points(self):
        for index in tqdm(range(len(self.point_clouds))):
            current_point_cloud = self.point_clouds[index]
            current_label_cloud = self.point_cloud_labels[index]
            current_labels = self.all_labels[index]
            n_points = len(current_point_cloud)
            # Randomly sampling respective indices
            sampled_indices = random.sample(
                list(range(n_points)), self.n_sampled_points
            )
            # Sampling points corresponding to sampled indices
            sampled_point_cloud = np.array(
                [current_point_cloud[i] for i in sampled_indices]
            )
            # Sampling corresponding one-hot encoded labels
            sampled_label_cloud = np.array(
                [current_label_cloud[i] for i in sampled_indices]
            )
            # Sampling corresponding labels for visualization
            sampled_labels = np.array([current_labels[i] for i in sampled_indices])
            # Normalizing sampled point cloud
            normalized_point_cloud = sampled_point_cloud - np.mean(
                sampled_point_cloud, axis=0
            )
            normalized_point_cloud /= np.max(
                np.linalg.norm(normalized_point_cloud, axis=1)
            )
            self.point_clouds[index] = normalized_point_cloud
            self.point_cloud_labels[index] = sampled_label_cloud
            self.all_labels[index] = sampled_labels

    def load_data(self):
        """
        Function to load the ShapeNet Core Point Cloud and Label Cloud into the primary memory.
        """
        points_dir = os.path.join(
            self.dataset_path,
            "{}/points".format(self.metadata[self.object_category]["directory"]),
        )
        labels_dir = os.path.join(
            self.dataset_path,
            "{}/points_label".format(self.metadata[self.object_category]["directory"]),
        )
        points_files = glob(os.path.join(points_dir, "*.pts"))
        for point_file in tqdm(points_files):
            point_cloud = np.loadtxt(point_file)
            file_id = point_file.split("/")[-1].split(".")[0]
            label_data, num_labels = {}, 0
            for label in self.labels:
                label_file = os.path.join(labels_dir, label, file_id + ".seg")
                if os.path.exists(label_file):
                    label_data[label] = np.loadtxt(label_file).astype("float32")
                    num_labels = len(label_data[label])
            try:
                label_map = ["none"] * num_labels
                for label in self.labels:
                    for i, data in enumerate(label_data[label]):
                        label_map[i] = label if data == 1 else label_map[i]
                label_data = [
                    self.labels.index(label) if label != "none" else len(self.labels)
                    for label in label_map
                ]
                label_cloud = tf.keras.utils.to_categorical(
                    label_data, num_classes=len(self.labels) + 1
                )
                self.point_clouds.append(point_cloud)
                self.point_cloud_labels.append(label_cloud)
                self.all_labels.append(label_map)
                self.point_cloud_dataframes.append(
                    pd.DataFrame(
                        data={
                            "x": point_cloud[:, 0],
                            "y": point_cloud[:, 1],
                            "z": point_cloud[:, 2],
                            "label": label_map,
                        }
                    )
                )
            except KeyError:
                # Use point cloud files without labels as test data
                self.test_point_clouds.append(point_cloud)

    def visualize_data_plotly(self, index):
        """
        Function to visualize the point cloud and corresponding labels using Plotly.
        """
        fig = px.scatter_3d(
            self.point_cloud_dataframes[index],
            x="x",
            y="y",
            z="z",
            color="label",
            labels={"label": "Label"},
            color_discrete_sequence=self.colors,
            category_orders={"label": self.labels},
        )
        fig.show()

    def visualize_data_plt(self, index):
        """
        Function to visualize the point cloud and corresponding labels using Matplotlib.
        """
        df = self.point_cloud_dataframes[index]
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection="3d")
        for index, label in enumerate(self.labels):
            c_df = df[df["label"] == label]
            try:
                ax.scatter(
                    c_df["x"],
                    c_df["y"],
                    c_df["z"],
                    label=label,
                    alpha=0.5,
                    c=self.colors[index],
                )
            except IndexError:
                pass
        ax.legend()
        plt.show()

    def _load_data(self, point_cloud, label_cloud):
        point_cloud.set_shape([self.n_sampled_points, 3])
        label_cloud.set_shape([self.n_sampled_points, len(self.labels) + 1])
        return point_cloud, label_cloud

    def _generate_dataset(self, point_clouds, label_clouds, batch_size: int):
        dataset = tf.data.Dataset.from_tensor_slices((point_clouds, label_clouds))
        dataset = dataset.map(self._load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size)
        return dataset

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
        self._sample_points()
        split_index = int(len(self.point_clouds) * (1 - val_split))
        train_point_clouds = self.point_clouds[:split_index]
        train_point_cloud_labels = self.point_cloud_labels[:split_index]
        val_point_clouds = self.point_clouds[split_index:]
        val_point_cloud_labels = self.point_cloud_labels[split_index:]
        train_dataset = self._generate_dataset(
            train_point_clouds, train_point_cloud_labels, batch_size
        )
        val_dataset = self._generate_dataset(
            val_point_clouds, val_point_cloud_labels, batch_size
        )
        return train_dataset, val_dataset

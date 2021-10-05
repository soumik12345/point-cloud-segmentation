import tensorflow as tf


def apply_jitter(point_cloud_batch, label_cloud_batch):
    # Jitter point and label clouds.
    noise = tf.random.uniform(
        tf.shape(label_cloud_batch), -0.005, 0.005, dtype=tf.float64
    )
    point_cloud_batch += noise[:, :, :3]
    label_cloud_batch += tf.cast(noise, tf.float32)
    return point_cloud_batch, label_cloud_batch

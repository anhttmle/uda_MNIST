import tensorflow as tf
from generator.image import MNISTSupDataGenerator, MNISTUnSupDataGenerator
import numpy as np


def iterator_to_generator(iterator):
    def get_generator():
        for item in iterator:
            yield item

    return get_generator


def build_mnist_dataset(n_sup=1000):
    train_set, test_set = tf.keras.datasets.mnist.load_data()
    train_sup_generator = MNISTSupDataGenerator(
        images=train_set[0][:n_sup],
        labels=train_set[1][:n_sup],
        target_size=(28, 28),
        n_channel=1
    )

    train_sup_dataset = tf.data.Dataset.from_generator(
        generator=iterator_to_generator(train_sup_generator),
        output_types=(tf.float32, tf.int32),
        output_shapes=((28, 28, 1), ())
    ).shuffle(buffer_size=1000).batch(batch_size=16).repeat()

    train_unsup_generator = MNISTUnSupDataGenerator(
        images=train_set[0],
        target_size=(28, 28),
        n_channel=1,
        n_apply_transform=5,
        magnitude=5
    )

    train_unsup_dataset = tf.data.Dataset.from_generator(
        generator=iterator_to_generator(train_unsup_generator),
        output_types=(tf.float32, tf.float32),
        output_shapes=((28, 28, 1), (28, 28, 1))
    ).shuffle(buffer_size=1000).batch(32).repeat()

    test_sup_generator = MNISTSupDataGenerator(
        images=test_set[0],
        labels=test_set[1],
        target_size=(28, 28),
        n_channel=1
    )

    test_sup_dataset = tf.data.Dataset.from_generator(
        generator=iterator_to_generator(test_sup_generator),
        output_types=(tf.float32, tf.int32),
        output_shapes=((28, 28, 1), ())
    ).batch(batch_size=32)

    return train_sup_dataset, train_unsup_dataset, test_sup_dataset


# train_sup_dataset, train_unsup_dataset, test_sup_dataset = build_mnist_dataset()

@tf.function
def train(dataset):
    ds = iter(dataset)
    for i in tf.range(tf.constant(2)):
        sup_images, sup_labels = next(ds)
        tf.print(sup_labels)

    pass


# train(train_sup_dataset)
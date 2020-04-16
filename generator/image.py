import tensorflow.keras as keras

import numpy as np
from PIL import Image

from data import normalize
from augmentation.transform import DERandAugment


class MNISTSupDataGenerator(keras.utils.Sequence):
    def __init__(
            self,
            images,
            labels,
            target_size=(28, 28),
            n_channel=1,
    ):
        assert len(images) == len(labels), "Number of image must equal number of label"
        self.images = images
        self.labels = labels
        self.target_size = target_size
        self.n_channel = n_channel
        self.batch_size = 1
        return

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, idx):
        batch = (
            self.images[idx * self.batch_size:(idx + 1) * self.batch_size],
            self.labels[idx * self.batch_size:(idx + 1) * self.batch_size],
        )

        n_data = len(batch[0])

        batch_images = np.zeros(shape=(n_data,) + self.target_size + (self.n_channel,))
        batch_label = np.zeros(shape=n_data, dtype=np.int)

        for i in range(n_data):
            image = batch[0][i]
            image = np.stack([image, image, image], axis=-1)
            image = Image.fromarray(image)

            image, annotations = normalize.resize(image=image, annotations=[], target_size=self.target_size)
            image, annotations = normalize.scale_to_unit(image=image, annotations=[])
            image = np.mean(image, axis=-1, keepdims=True)

            batch_images[i] = image.astype(np.float)
            batch_label[i] = batch[1][i]

        if self.batch_size == 1:
            batch_images = np.reshape(
                batch_images,
                newshape=(batch_images.shape[1], batch_images.shape[2], batch_images.shape[3])
            )
            batch_label = np.reshape(batch_label, newshape=())

        return batch_images, batch_label

    def on_epoch_end(self):
        print("Supervise: End of epoch")
        return


class MNISTUnSupDataGenerator(keras.utils.Sequence):
    def __init__(
            self,
            images,
            target_size=(28, 28),
            n_channel=1,
            n_apply_transform=5,
            magnitude=3,
    ):
        self.images = images
        self.batch_size = 1
        self.target_size = target_size
        self.n_channel = n_channel
        self.augmenter = DERandAugment(
            n_apply_transform=n_apply_transform,
            magnitude=magnitude
        )
        return

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]

        n_data = len(images)

        batch_images = np.zeros(shape=(n_data,) + self.target_size + (self.n_channel,))
        batch_images_aug = np.zeros(shape=(n_data,) + self.target_size + (self.n_channel,))

        for i in range(n_data):
            image = images[i]
            image_aug = np.stack([image, image, image], axis=-1)
            image_aug = Image.fromarray(image_aug)

            image_aug, annotations = normalize.resize(image=image_aug, annotations=[], target_size=self.target_size)
            image_aug, annotations = self.augmenter.transform(image=image_aug, annotations=None)

            image_aug, annotations = normalize.scale_to_unit(image=image_aug, annotations=[])
            image_aug = np.mean(image_aug, axis=-1, keepdims=True)

            batch_images[i] = np.expand_dims(image, axis=-1).astype(np.float)
            batch_images_aug[i] = image_aug.astype(np.float)

        if self.batch_size == 1:
            batch_images = np.reshape(
                batch_images,
                newshape=(batch_images.shape[1], batch_images.shape[2], batch_images.shape[3])
            )

            batch_images_aug = np.reshape(
                batch_images_aug,
                newshape=(batch_images_aug.shape[1], batch_images_aug.shape[2], batch_images_aug.shape[3])
            )

        return batch_images, batch_images_aug

    def on_epoch_end(self):
        print("UnSupervise: End of epoch")
        return


class CIFAR10SupDataGenerator(keras.utils.Sequence):
    def __init__(
            self,
            images,
            labels,
            target_size=(32, 32),
            n_channel=3,
    ):
        assert len(images) == len(labels), "Number of image must equal number of label"
        self.images = images
        self.labels = labels
        self.target_size = target_size
        self.n_channel = n_channel
        self.batch_size = 1
        return

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, idx):
        batch = (
            self.images[idx * self.batch_size:(idx + 1) * self.batch_size],
            self.labels[idx * self.batch_size:(idx + 1) * self.batch_size],
        )

        n_data = len(batch[0])

        batch_images = np.zeros(shape=(n_data,) + self.target_size + (self.n_channel,))
        batch_label = np.zeros(shape=n_data, dtype=np.int)

        for i in range(n_data):
            image = batch[0][i]
            image = Image.fromarray(image)

            image, annotations = normalize.resize(image=image, annotations=[], target_size=self.target_size)
            image, annotations = normalize.scale_to_unit(image=image, annotations=[])

            batch_images[i] = image.astype(np.float)
            batch_label[i] = batch[1][i]

        if self.batch_size == 1:
            batch_images = np.reshape(
                batch_images,
                newshape=(batch_images.shape[1], batch_images.shape[2], batch_images.shape[3])
            )
            batch_label = np.reshape(batch_label, newshape=())

        return batch_images, batch_label

    def on_epoch_end(self):
        print("Supervise: End of epoch")
        return


class CIFAR10UnSupDataGenerator(keras.utils.Sequence):
    def __init__(
            self,
            images,
            target_size=(32, 32),
            n_channel=3,
            n_apply_transform=5,
            magnitude=3,
    ):
        self.images = images
        self.batch_size = 1
        self.target_size = target_size
        self.n_channel = n_channel
        self.augmenter = DERandAugment(
            n_apply_transform=n_apply_transform,
            magnitude=magnitude
        )
        return

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]

        n_data = len(images)

        batch_images = np.zeros(shape=(n_data,) + self.target_size + (self.n_channel,))
        batch_images_aug = np.zeros(shape=(n_data,) + self.target_size + (self.n_channel,))

        for i in range(n_data):
            image = images[i]
            image_aug = Image.fromarray(image)

            image_aug, annotations = normalize.resize(image=image_aug, annotations=[], target_size=self.target_size)
            image_aug, annotations = self.augmenter.transform(image=image_aug, annotations=None)

            image_aug, annotations = normalize.scale_to_unit(image=image_aug, annotations=[])

            batch_images_aug[i] = image_aug.astype(np.float)

        if self.batch_size == 1:
            batch_images = np.reshape(
                batch_images,
                newshape=(batch_images.shape[1], batch_images.shape[2], batch_images.shape[3])
            )

            batch_images_aug = np.reshape(
                batch_images_aug,
                newshape=(batch_images_aug.shape[1], batch_images_aug.shape[2], batch_images_aug.shape[3])
            )

        return batch_images, batch_images_aug

    def on_epoch_end(self):
        print("UnSupervise: End of epoch")
        return

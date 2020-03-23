import tensorflow as tf
import tensorflow.keras as keras

from generator.image import UnSupDataGenerator, SupDataGenerator
from losses.uda import compute_uda_loss

import sys


def get_data():
    train_set, test_set = keras.datasets.mnist.load_data()
    print(len(set(train_set[1][:100])))

    train_unsup_gen = UnSupDataGenerator(
        images=train_set[0][100:],
    )

    train_sup_gen = SupDataGenerator(
        images=train_set[0][:100],
        labels=train_set[1][:100],
    )

    test_gen = SupDataGenerator(
        images=test_set[0],
        labels=test_set[1]
    )

    return (train_sup_gen, train_unsup_gen), test_gen


def build_model():
    x = keras.layers.Input(shape=(28, 28, 1))
    mid = keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(x)
    mid = keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(mid)
    mid = keras.layers.BatchNormalization()(mid)
    mid = keras.layers.Activation(activation="relu")(mid)
    mid = keras.layers.MaxPool2D()(mid)
    mid = keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(mid)
    mid = keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(mid)
    mid = keras.layers.BatchNormalization()(mid)
    mid = keras.layers.Activation(activation="relu")(mid)
    mid = keras.layers.MaxPool2D()(mid)
    mid = keras.layers.Flatten()(mid)
    y_logit = keras.layers.Dense(10, name='logit')(mid)

    return keras.models.Model(
        inputs=x,
        outputs=y_logit
    )


def train(
        model,
        train_sup_gen,
        train_unsup_gen,
        test_sup_gen,
        n_step=10000,
):
    optimizer = tf.optimizers.Adam()
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=["acc"]
    )

    for step in range(n_step):
        sup_images, sup_labels = train_sup_gen[step % len(train_sup_gen)]
        unsup_images, unsup_images_aug = train_unsup_gen[step % len(train_unsup_gen)]

        with tf.GradientTape() as tape:
            loss = compute_uda_loss(
                model=model,
                sup_images=sup_images,
                sup_labels=sup_labels,
                unsup_images=unsup_images,
                unsup_images_aug=unsup_images_aug,
                current_step=step,
                total_step=n_step,
                n_classes=10,
                tsa=True,
            )

        grads = tape.gradient(
            loss, model.trainable_variables
        )

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        sys.stdout.write("\rStep: {},         Loss: {}".format(
            optimizer.iterations.numpy(),
            loss.numpy())
        )

        if step % 200 == 0:
            print()
            evaluate(model=model, dataset=test_sup_gen)
            print()
            # model.evaluate(test_sup_gen)

    return model


def evaluate(
        model,
        dataset
):
    acc_fn = keras.metrics.SparseCategoricalAccuracy()
    acc_fn.reset_states()

    for i in range(len(dataset)):
        sup_images, sup_labels = dataset[i]
        acc = acc_fn(sup_labels, tf.nn.softmax(model(sup_images), axis=-1))
        sys.stdout.write("\rACC: {} = {}/{}".format(acc.numpy(), acc_fn.total.numpy(), acc_fn.count.numpy()))


if __name__ == "__main__":
    train_gen, test_gen = get_data()
    train_sup_gen, train_unsup_gen = train_gen

    model = build_model()

    model = train(
        model=model,
        train_sup_gen=train_sup_gen,
        train_unsup_gen=train_unsup_gen,
        test_sup_gen=test_gen,
        n_step=10000,
    )




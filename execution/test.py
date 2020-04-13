import tensorflow as tf
import tensorflow.keras as keras

from losses.uda import compute_uda_loss
import models as models
import data.dataset as dataset


import sys


def get_data():
    return dataset.build_mnist_dataset(n_sup=1000)


@tf.function
def train_with_uda(
        train_sup_dataset,
        train_unsup_dataset,
        model,
        optimizer,
        n_step,
):
    get_unsup_data = iter(train_unsup_dataset)
    get_sup_data = iter(train_sup_dataset)

    for step in tf.range(n_step):
        sup_images, sup_labels = next(get_sup_data)
        unsup_images, unsup_images_aug = next(get_unsup_data)

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

        tf.print(
            "Step: ", step + 1, ", Loss: ", loss,
            output_stream=sys.stdout
        )


@tf.function
def evaluate(
        model,
        eval_ds,
        metric_fn,
):
    metric_fn.reset_states()
    for sup_images, sup_labels in eval_ds:
        metric_fn(sup_labels, model(sup_images))


if __name__ == "__main__":
    train_sup_ds, train_unsup_ds, test_sup_ds = get_data()

    baseline_model = models.get_baseline_model()
    opt = tf.optimizers.Adam()

    train_with_uda(
        train_sup_dataset=train_sup_ds,
        train_unsup_dataset=train_unsup_ds,
        model=baseline_model,
        optimizer=opt,
        n_step=tf.constant(1000)
    )

    acc_fn = keras.metrics.SparseCategoricalAccuracy()

    print("ACC: {}".format(acc_fn.result()))

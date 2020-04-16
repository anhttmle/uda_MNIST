import tensorflow as tf
import tensorflow.keras as keras

from losses.uda import compute_uda_loss, compute_sup_loss
import models as models
import data.dataset as dataset


import sys


def get_data():
    return dataset.build_cifar10_dataset(n_sup=1000)


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

        if step % 1000 == 0:
            tf.print(
                "Step: ", step + 1, ", Loss: ", loss,
                output_stream=sys.stdout
            )


@tf.function
def train(
    train_sup_dataset,
    model,
    optimizer,
    n_step,
):
    get_sup_data = iter(train_sup_dataset)

    for step in tf.range(n_step):
        sup_images, sup_labels = next(get_sup_data)

        with tf.GradientTape() as tape:
            sup_logits = model(sup_images)
            loss = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(
                    y_true=sup_labels,
                    y_pred=sup_logits,
                    from_logits=True
                )
            )

        grads = tape.gradient(
            loss, model.trainable_variables
        )

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 1000 == 0:
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

    baseline_model = models.get_baseline_model(width=32, height=32, n_channel=3)
    opt = tf.optimizers.Adam()

    # writer = tf.summary.create_file_writer(logdir="logs/func/temp")

    train_with_uda(
        train_sup_dataset=train_sup_ds,
        train_unsup_dataset=train_unsup_ds,
        model=baseline_model,
        optimizer=opt,
        n_step=tf.constant(100000, name="n_step")
    )



    # train(
    #     train_sup_dataset=train_sup_ds,
    #     model=baseline_model,
    #     optimizer=opt,
    #     n_step=tf.constant(10000)
    # )

    acc_fn = keras.metrics.SparseCategoricalAccuracy()
    # tf.summary.trace_on(graph=True, profiler=False)
    evaluate(
        model=baseline_model,
        eval_ds=test_sup_ds,
        metric_fn=acc_fn
    )
    # with writer.as_default():
    #     tf.summary.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         # profiler_outdir=log
    #     )

    print("ACC: {}".format(acc_fn.result()))


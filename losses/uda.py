import tensorflow as tf

from config.enum import TSA_Schedule
from metrics import kl_divergence_with_logits, entropy_with_logits
from utils.progress import tsa_threshold


def compute_sup_loss(
        sup_labels,
        sup_logits,
        current_step,
        total_steps,
        n_classes,
        tsa=False
):
    sup_loss = tf.losses.sparse_categorical_crossentropy(
        y_true=sup_labels,
        y_pred=sup_logits,
        from_logits=True
    )

    avg_sup_loss = tf.reduce_mean(sup_loss)

    if tsa:
        start_threshold = 1. / n_classes
        end_threshold = 1.
        current_threshold = tsa_threshold(
            schedule=TSA_Schedule.LOG,
            current_step=current_step,
            total_steps=total_steps,
            start_threshold=start_threshold,
            end_threshold=end_threshold,
            scale=tf.constant(5, dtype=tf.float32)
        )

        one_hot_sup_labels = tf.one_hot(
            indices=sup_labels,
            depth=n_classes,
            dtype=tf.float32
        )

        sup_probs = tf.nn.softmax(sup_logits, axis=-1)

        correct_label_probs = tf.reduce_sum(
            one_hot_sup_labels * sup_probs,
            axis=-1
        )

        # Example which is higher than threshold will be removed
        larger_than_threshold = tf.greater(
            x=correct_label_probs,
            y=current_threshold
        )

        loss_mask = 1 - tf.cast(larger_than_threshold, tf.float32)
        loss_mask = tf.stop_gradient(loss_mask)

        sup_loss = sup_loss * loss_mask
        n_example = tf.reduce_sum(loss_mask)
        avg_sup_loss = tf.reduce_sum(sup_loss) / tf.maximum(n_example, 1)

    return sup_loss, avg_sup_loss


def compute_augment_loss(
        origin_logits,
        augment_logits,
        uda_threshold=-1,
        coeff=1
):
    augment_loss = kl_divergence_with_logits(
        p_logits=tf.stop_gradient(origin_logits),
        q_logits=augment_logits
    )

    if uda_threshold != -1:
        origin_prob = tf.nn.softmax(origin_logits, axis=-1)
        origin_max_prob = tf.reduce_max(origin_prob, axis=-1)

        # Mask out all prediction which is higher than uda_threshold
        loss_mask = tf.greater(x=origin_max_prob, y=uda_threshold)
        loss_mask = tf.cast(loss_mask, dtype=tf.float32)
        # Not propagate the mask
        loss_mask = tf.stop_gradient(loss_mask)
        augment_loss = augment_loss * loss_mask

    augment_loss = tf.reduce_mean(augment_loss) * coeff

    return augment_loss


def compute_entropy_loss(
        origin_logits,
        coeff=0
):
    if coeff <= 0:
        return 0

    origin_entropy = entropy_with_logits(origin_logits)
    entropy_loss = tf.reduce_mean(origin_entropy)
    return entropy_loss * coeff


def compute_total_loss(
        sup_labels,
        sup_logits,
        origin_logits,
        augment_logits,
        current_step,
        total_step,
        n_classes,
        tsa=False,
        batch_size_ratio=0,
        uda_softmax_temp=-1,
        uda_threshold=-1,
        augment_coeff=1,
        entropy_coeff=0
):
    sup_loss, avg_sup_loss = compute_sup_loss(
        sup_labels=sup_labels,
        sup_logits=sup_logits,
        current_step=current_step,
        total_steps=total_step,
        n_classes=n_classes,
        tsa=tsa
    )

    total_loss = avg_sup_loss

    if batch_size_ratio >= 0:
        if uda_softmax_temp != -1:
            origin_logits = origin_logits/uda_softmax_temp

        augment_loss = compute_augment_loss(
            origin_logits=origin_logits,
            augment_logits=augment_logits,
            uda_threshold=uda_threshold,
            coeff=augment_coeff
        )

        total_loss += augment_loss

        entropy_loss = compute_entropy_loss(
            origin_logits=origin_logits,
            coeff=entropy_coeff
        )

        total_loss += entropy_loss

    return total_loss


def compute_uda_loss(
        model,
        sup_images,
        sup_labels,
        unsup_images,
        unsup_images_aug,
        current_step,
        total_step,
        n_classes,
        tsa=False,
        batch_size_ratio=0,
        uda_softmax_temp=-1,
        uda_threshold=-1,
        augment_coeff=1,
        entropy_coeff=0
):

    return compute_total_loss(
        sup_labels=sup_labels,
        sup_logits=model(sup_images),
        origin_logits=model(unsup_images),
        augment_logits=model(unsup_images_aug),
        current_step=current_step,
        total_step=total_step,
        n_classes=n_classes,
        tsa=tsa,
        batch_size_ratio=batch_size_ratio,
        uda_softmax_temp=uda_softmax_temp,
        uda_threshold=uda_threshold,
        augment_coeff=augment_coeff,
        entropy_coeff=entropy_coeff
    )
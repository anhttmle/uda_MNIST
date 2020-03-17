from config.enum import TSA_Schedule

import tensorflow as tf


def tsa_threshold(schedule, current_step, total_steps, start_threshold, end_threshold):
    progress_ratio = tf.cast(current_step/total_steps, dtype=tf.float32)
    if schedule == TSA_Schedule.LOG:
        scale = 5
        threshold_cof = 1 - tf.exp((-progress_ratio) * scale)
    elif schedule == TSA_Schedule.LINEAR:
        threshold_cof = progress_ratio
    elif schedule == TSA_Schedule.EXP:
        scale = 5
        threshold_cof = tf.exp((progress_ratio - 1) * scale)
    else:
        raise TypeError("Schedule can only be TSA_Schedule.LOG/LINEAR/EXP")

    return threshold_cof * (end_threshold - start_threshold) + start_threshold
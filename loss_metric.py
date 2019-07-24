import tensorflow as tf


def score_loss(label_list, result_list):
    print("score list shape", label_list, result_list)
    distance = tf.abs(result_list - label_list)
    mask = tf.cast(tf.less(distance, 1.0), tf.float32)
    loss = mask * (0.5 * distance ** 2) + (1.0 - mask) * (distance - 0.5)
    loss = tf.reduce_mean(loss)

    return loss


def siam_loss(label_list, result_list):
    print("siam list shape", label_list, result_list)
    distance = tf.abs(result_list - label_list)
    mask = tf.cast(tf.less(distance, 1.0), tf.float32)
    loss = mask * (0.5 * distance ** 2) + (1.0 - mask) * (distance - 0.5)
    loss = tf.reduce_mean(loss)

    return loss


def locate_loss(label_list, result_list):
    print("locate list shape", label_list, result_list)

    # focal loss for binary task/sigmoid
    alpha = 0.25
    gamma = 2

    focal_loss = tf.multiply(-label_list, 1 - alpha) * tf.pow(1 - result_list, gamma) * tf.log(result_list) - \
                 tf.multiply(1 - label_list, alpha) * tf.pow(result_list, gamma) * tf.log(1 - result_list)

    focal_loss = tf.reduce_mean(focal_loss)

    return focal_loss


def score_metric(label_list, result_list):
    print("score metric list shape", label_list, result_list)
    abs_errors = tf.abs(label_list - result_list)
    # mae = tf.reduce_mean(tf.abs(label_list - result_list))
    return abs_errors


def locate_metric(label_list, result_list):
    print("locate metric list shape", label_list, result_list)
    mean_pixel_error = tf.reduce_mean(tf.abs(label_list - result_list))
    return mean_pixel_error

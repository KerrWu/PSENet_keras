import tensorflow as tf
from keras.layers import Conv2D, Add, Multiply
from keras import regularizers


def score_refine_module(input_feature_map, map_name=None):
    global weight_decay

    with tf.variable_scope("score_refine_module", reuse=tf.AUTO_REUSE):
        x = Conv2D(256, (1, 1), activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(input_feature_map)

        # score head:
        score_map_s1 = Conv2D(256, (1, 1), padding='same', activation="relu",
                              kernel_regularizer=regularizers.l2(weight_decay))(x)
        score_map_s2 = Conv2D(256, (3, 3), padding='same', activation="relu",
                              kernel_regularizer=regularizers.l2(weight_decay))(x)
        score_map_s1s2 = Add()([score_map_s1, score_map_s2])

        score_map_s1 = Conv2D(256, (1, 1), padding='same', activation="relu",
                              kernel_regularizer=regularizers.l2(weight_decay))(score_map_s1s2)
        score_map_s2 = Conv2D(256, (3, 3), padding='same', activation="relu",
                              kernel_regularizer=regularizers.l2(weight_decay))(score_map_s1s2)
        score_map_s1s2 = Add()([score_map_s1, score_map_s2])

        score_map_s1 = Conv2D(256, (1, 1), padding='same', activation="relu",
                              kernel_regularizer=regularizers.l2(weight_decay))(score_map_s1s2)
        score_map_s2 = Conv2D(256, (3, 3), padding='same', activation="relu",
                              kernel_regularizer=regularizers.l2(weight_decay))(score_map_s1s2)
        score_map_s1s2 = Add()([score_map_s1, score_map_s2])

        score_map_s1 = Conv2D(256, (1, 1), padding='same', activation="relu",
                              kernel_regularizer=regularizers.l2(weight_decay))(score_map_s1s2)
        score_map_s2 = Conv2D(256, (3, 3), padding='same', activation="relu",
                              kernel_regularizer=regularizers.l2(weight_decay))(score_map_s1s2)
        score_map_s1s2 = Add()([score_map_s1, score_map_s2])

        score_map = Conv2D(5, (1, 1), padding='same', activation="relu",
                           kernel_regularizer=regularizers.l2(weight_decay))(score_map_s1s2)

        # locate head
        locate_head = Conv2D(256, (1, 1), padding='same', activation="relu",
                             kernel_regularizer=regularizers.l2(weight_decay))(x)
        locate_head = Conv2D(256, (1, 1), padding='same', activation="relu",
                             kernel_regularizer=regularizers.l2(weight_decay))(locate_head)
        locate_head = Conv2D(256, (1, 1), padding='same', activation="relu",
                             kernel_regularizer=regularizers.l2(weight_decay))(locate_head)
        locate_head = Conv2D(256, (1, 1), padding='same', activation="relu",
                             kernel_regularizer=regularizers.l2(weight_decay))(locate_head)

        if map_name == None:
            locate_map = Conv2D(1, (1, 1), padding='same', activation="sigmoid",
                                kernel_regularizer=regularizers.l2(weight_decay))(locate_head)
        else:
            locate_map = Conv2D(1, (1, 1), padding='same', activation="sigmoid", name=map_name + "_locate",
                                kernel_regularizer=regularizers.l2(weight_decay))(locate_head)

        refined_map = Multiply()([score_map, locate_map])

    return refined_map, locate_map

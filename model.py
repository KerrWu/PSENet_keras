import keras
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, Add, GlobalAveragePooling2D, Multiply, \
    Concatenate, MaxPooling2D
from keras import regularizers

from SR_Module import score_refine_module


def combine_siamese_results(output_score_map_a, output_score_map_b):
    # 该函数输出5*3个值，A网络的5个分数，B网络的5个分数以及AB两网络的5个分数差距

    # score A:
    A_vector = Concatenate()([GlobalAveragePooling2D()(output_score_map_a[i]) for i in range(5)])
    A_score = Dense(5, activation=None, name="scoreA", kernel_regularizer=regularizers.l2(weight_decay))(A_vector)

    # score B:
    B_vector = Concatenate()([GlobalAveragePooling2D()(output_score_map_b[i]) for i in range(5)])
    B_score = Dense(5, activation=None, name="scoreB", kernel_regularizer=regularizers.l2(weight_decay))(B_vector)

    # score gap:
    siamese_map = [Add()([output_score_map_a[i], output_score_map_b[i]]) for i in range(5)]
    siamese_vector = Concatenate()([GlobalAveragePooling2D()(siamese_map[i]) for i in range(5)])
    siamese_score = Dense(5, activation=None, name="scoreSiam", kernel_regularizer=regularizers.l2(weight_decay))(
        siamese_vector)

    return A_score, B_score, siamese_score


def PSENet():
    global height, width, weight_decay
    # define single network
    with tf.device('/cpu:0'):
        image_input = Input(shape=(800, 1024, 3), name="single_input")

        with tf.variable_scope("resnet50", reuse=tf.AUTO_REUSE):
            base_model = ResNet50(weights=None, input_shape=(800, 1024, 3), input_tensor=image_input, include_top=False)

            for layer in base_model.layers:
                if isinstance(layer, keras.layers.DepthwiseConv2D):
                    layer.add_loss(keras.regularizers.l2(weight_decay)(layer.depthwise_kernel))
                elif isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                    layer.add_loss(keras.regularizers.l2(weight_decay)(layer.kernel))
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.add_loss(keras.regularizers.l2(weight_decay)(layer.bias))

        with tf.variable_scope("FPN_SRM", reuse=tf.AUTO_REUSE):
            # p2 = base_model.get_layer("activation_10").output
            # p2_score_map, p2_locate_map = score_refine_module(p2, "p2")

            p3 = base_model.get_layer("activation_22").output
            p3_score_map, p3_locate_map = score_refine_module(p3, "p3")

            p4 = base_model.get_layer("activation_40").output
            p4_score_map, p4_locate_map = score_refine_module(p4, "p4")

            p5 = base_model.get_layer("activation_49").output
            p5_score_map, p5_locate_map = score_refine_module(p5, "p5")

            p6 = Conv2D(256, (1, 1), padding='same', activation="relu",
                        kernel_regularizer=regularizers.l2(weight_decay))(
                base_model.output)
            p6 = Conv2D(256, (3, 3), padding='same', activation="relu",
                        kernel_regularizer=regularizers.l2(weight_decay))(
                p6)
            p6 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None)(p6)
            p6_score_map, p6_locate_map = score_refine_module(p6, "p6")

            p7 = Conv2D(256, (1, 1), padding='same', activation="relu",
                        kernel_regularizer=regularizers.l2(weight_decay))(
                p6)
            p7 = Conv2D(256, (3, 3), padding='same', activation="relu",
                        kernel_regularizer=regularizers.l2(weight_decay))(
                p7)
            p7 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None)(p7)
            p7_score_map, p7_locate_map = score_refine_module(p7, "p7")

    single_model = Model(inputs=image_input,
                         outputs=[p3_score_map, p4_score_map, p5_score_map, p6_score_map, p7_score_map, p3_locate_map,
                                  p4_locate_map, p5_locate_map, p6_locate_map, p7_locate_map], name="subnetwork")

    # define siamese network
    with tf.device('/cpu:0'):
        input_a = Input(shape=(height, width, 3), name="input_a")
        input_b = Input(shape=(height, width, 3), name="input_b")

    output_a = single_model(input_a)
    output_b = single_model(input_b)

    with tf.device('/cpu:0'):
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            A_score, B_score, siamese_score = combine_siamese_results(output_a[:5], output_b[:5])

    # siamese模型的总输出包括：
    # A_score 5个分, B_score 5个分, siamese_score 5个分, A_locate_map x 5, B_locate_map x 5
    # 5个分的顺序是area, ery, sca, ind, pasi

    output_list = [A_score, B_score, siamese_score]

    output_list.extend(output_a[5:])
    output_list.extend(output_b[5:])
    print(len(output_list))

    siamese_model = Model(inputs=[input_a, input_b], outputs=output_list, name="siamese")

    return siamese_model

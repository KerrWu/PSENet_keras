import keras
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, Add, GlobalAveragePooling2D, Multiply, \
    Concatenate, MaxPooling2D
from keras import regularizers
from SR_Module import score_refine_module
from global_var import myModelConfig
from keras import backend as K
from keras.layers import Layer, Lambda

def combine_siamese_results(output_score_map_a, output_score_map_b):
    # 该函数输出5*3个值，A网络的5个分数，B网络的5个分数以及AB两网络的5个分数差距

    # score A:
    A_vector = Concatenate()([GlobalAveragePooling2D()(output_score_map_a[i]) for i in range(5)])
    A_score = Dense(5, activation=None, name="scoreA", kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(A_vector)

    # score B:
    B_vector = Concatenate()([GlobalAveragePooling2D()(output_score_map_b[i]) for i in range(5)])
    B_score = Dense(5, activation=None, name="scoreB", kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(B_vector)

    # score gap:
    siamese_map = [Add()([output_score_map_a[i], output_score_map_b[i]]) for i in range(5)]
    siamese_vector = Concatenate()([GlobalAveragePooling2D()(siamese_map[i]) for i in range(5)])
    siamese_score = Dense(5, activation=None, name="scoreSiam", kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(siamese_vector)

    return A_score, B_score, siamese_score

def fpn_combine(pair):
    deep = pair[0]
    shallow = pair[1]
    shallow_shape = tf.shape(shallow)
    deep_up = tf.image.resize_nearest_neighbor(deep, [shallow_shape[1], shallow_shape[2]])
    shallow = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(shallow)
    combine_map = shallow + deep_up
    combine_out = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(combine_map)
    return tf.convert_to_tensor((combine_map, combine_out))


def PSENet(myModelConfig):

    # define single network
    with tf.device('/cpu:0'):

        image_input = Input(shape=(myModelConfig.img_height, myModelConfig.img_width, 3), name="single_input")

        with tf.variable_scope("resnet50", reuse=tf.AUTO_REUSE):
            base_model = ResNet50(weights="imagenet", input_shape=(800, 1024, 3), input_tensor=image_input,
                                  include_top=False)

            for layer in base_model.layers:
                if isinstance(layer, keras.layers.DepthwiseConv2D):
                    layer.add_loss(keras.regularizers.l2(myModelConfig.weight_decay)(layer.depthwise_kernel))
                elif isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                    layer.add_loss(keras.regularizers.l2(myModelConfig.weight_decay)(layer.kernel))
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.add_loss(keras.regularizers.l2(myModelConfig.weight_decay)(layer.bias))

        with tf.variable_scope("FPN_SRM", reuse=tf.AUTO_REUSE):
            # p2 = base_model.get_layer("activation_10").output
            # p2_score_map, p2_locate_map = score_refine_module(p2, "p2")

            # FPN
            p3 = base_model.get_layer("activation_22").output
            p4 = base_model.get_layer("activation_40").output
            p5 = base_model.get_layer("activation_49").output

            p6 = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(base_model.output)
            p6 = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p6)
            p6 = MaxPooling2D(pool_size=(2, 2), padding='same')(p6)

            p7 = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p6)
            p7 = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p7)
            p7 = MaxPooling2D(pool_size=(2, 2), padding='same')(p7)

            p6_map, p6 = Lambda(fpn_combine)([p7, p6])
            p5_map, p5 = Lambda(fpn_combine)([p6, p5])
            p4_map, p4 = Lambda(fpn_combine)([p5, p4])
            p3_map, p3 = Lambda(fpn_combine)([p4, p3])

            # p6_shape = tf.shape(p6)
            # p7_up = tf.image.resize_nearest_neighbor(p7, [p6_shape[1], p6_shape[2]])
            # p6 = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p6)
            # p6_map = p6 + p7_up
            # p6 = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p6_map)
            #
            # p5_shape = tf.shape(p5)
            # p6_up = tf.image.resize_nearest_neighbor(p6, [p5_shape[1], p5_shape[2]])
            # p5 = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p5)
            # p5_map = p5 + p6_up
            # p5 = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p5_map)
            #
            # p4_shape = tf.shape(p4)
            # p5_up = tf.image.resize_nearest_neighbor(p5, [p4_shape[1], p4_shape[2]])
            # p4 = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p4)
            # p4_map = p4 + p5_up
            # p4 = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p4_map)
            #
            # p3_shape = tf.shape(p3)
            # p4_up = tf.image.resize_nearest_neighbor(p4, [p3_shape[1], p3_shape[2]])
            # p3 = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p3)
            # p3_map = p3 + p4_up
            # p3 = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p3_map)

            p3_score_map, p3_locate_map = score_refine_module(p3, "p3")
            p4_score_map, p4_locate_map = score_refine_module(p4, "p4")
            p5_score_map, p5_locate_map = score_refine_module(p5, "p5")
            p6_score_map, p6_locate_map = score_refine_module(p6, "p6")
            p7_score_map, p7_locate_map = score_refine_module(p7, "p7")


            # p3 = base_model.get_layer("activation_22").output
            # p3_score_map, p3_locate_map = score_refine_module(p3, "p3")
            #
            # p4 = base_model.get_layer("activation_40").output
            # p4_score_map, p4_locate_map = score_refine_module(p4, "p4")
            #
            # p5 = base_model.get_layer("activation_49").output
            # p5_score_map, p5_locate_map = score_refine_module(p5, "p5")
            #
            # p6 = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(weight_decay))(base_model.output)
            # p6 = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(weight_decay))(p6)
            # p6 = MaxPooling2D(pool_size=(2, 2), padding='same')(p6)
            # p6_score_map, p6_locate_map = score_refine_module(p6, "p6")
            #
            # p7 = Conv2D(256, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(weight_decay))(p6)
            # p7 = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer='he_normal',
            #             kernel_regularizer=regularizers.l2(weight_decay))(p7)
            # p7 = MaxPooling2D(pool_size=(2, 2), padding='same')(p7)
            # p7_score_map, p7_locate_map = score_refine_module(p7, "p7")

    single_model = Model(inputs=image_input,
                         outputs=[p3_score_map, p4_score_map, p5_score_map, p6_score_map, p7_score_map, p3_locate_map,
                                  p4_locate_map, p5_locate_map, p6_locate_map, p7_locate_map], name="subnetwork")

    # define siamese network
    with tf.device('/cpu:0'):
        input_a = Input(shape=(myModelConfig.img_height, myModelConfig.img_width, 3), name="input_a")
        input_b = Input(shape=(myModelConfig.img_height, myModelConfig.img_width, 3), name="input_b")

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

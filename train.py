import os
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras import callbacks
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from global_var import myModelConfig
from model import PSENet
from data_generator import train_generator, valid_generator
from loss_metric import score_loss, siam_loss, locate_loss, score_metric, locate_metric
from self_callbacks import MyEarlyStop
from acc_opt import SGDAccumulate

os.environ["CUDA_VISIBLE_DEVICES"] = myModelConfig.availiable_gpus
steps_per_epoch_train = int(myModelConfig.num_train_examples_per_epoch // myModelConfig.batch_size)
steps_per_epoch_val = int(myModelConfig.num_val_examples_per_epoch // myModelConfig.batch_size)

train_gen = train_generator(myModelConfig.batch_size)
valid_gen = valid_generator()


siamese_model = PSENet(myModelConfig)

print(siamese_model.input)
print(siamese_model.output)
siamese_model.summary()

# parallel_model = siamese_model
parallel_model = multi_gpu_model(siamese_model, gpus=myModelConfig.num_gpus)

sgd_accu = SGDAccumulate(lr=myModelConfig.learning_rate, momentum=myModelConfig.momentum, nesterov=True,
                         accum_iters=myModelConfig.accu_num)
# sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=myModelConfig.learning_rate_decay_factor, patience=5,
                                        verbose=1, mode='min', cooldown=10, min_lr=0.00001)
my_early_stop = MyEarlyStop(siamese_model, myModelConfig.checkpoint_dir)
myTensorboard = callbacks.TensorBoard(log_dir=myModelConfig.summary_dir, histogram_freq=0, write_graph=False,
                                      write_images=True)

my_call_back = [my_early_stop, reduce_lr, myTensorboard]
parallel_model.compile(
    loss={"scoreA": score_loss, "scoreB": score_loss, "scoreSiam": siam_loss, "single_model": locate_loss},
    loss_weights={"scoreA": 0.1, "scoreB": 0.1, "scoreSiam": 0.05, "single_model": 1},
    optimizer=sgd_accu,
    metrics={"scoreA": score_metric, "scoreB": score_metric, "scoreSiam": score_metric, "single_model": locate_metric})

print("compiled")

history = None
try:
    history = parallel_model.fit_generator(generator=train_gen, epochs=myModelConfig.num_epochs, verbose=1,
                                           steps_per_epoch=steps_per_epoch_train,
                                           callbacks=my_call_back,
                                           validation_data=valid_gen,
                                           validation_steps=steps_per_epoch_val,
                                           max_queue_size=16,
                                           initial_epoch=0)

except KeyboardInterrupt:
    print("Early stop by user !")

except StopIteration:
    print("Training process finished !")

except:
    print("training process error")
    raise


finally:
    if history:
        def plot_train_history(history, train_metrics, val_metrics):
            plt.plot(history.history.get(train_metrics), '-o')
            plt.plot(history.history.get(val_metrics), '-o')
            plt.ylabel(train_metrics)
            plt.xlabel('Epochs')
            plt.legend(['train', 'validation'])


        plt.figure(figsize=(8, 4))
        plt.subplot(2, 2, 1)
        plot_train_history(history, 'loss', 'val_loss')
        plt.subplot(2, 2, 2)
        plot_train_history(history, 'scoreA_score_metric', 'val_scoreA_score_metric')
        plt.subplot(2, 2, 3)
        plot_train_history(history, 'scoreB_score_metric', 'val_scoreB_score_metric')
        plt.subplot(2, 2, 4)
        plot_train_history(history, 'scoreSiam_score_metric', 'val_scoreSiam_score_metric')

        plt.savefig(myModelConfig.history_file)

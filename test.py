import os
import tensorflow as tf
from acc_opt import SGDAccumulate
from keras.models import load_model
from data_generator import test_generator
from loss_metric import score_loss, siam_loss, locate_loss, score_metric, locate_metric

os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_path = "./experiments/checkpoints"

model_list = [elem for elem in os.listdir(model_path) if elem.endswith("h5")]
if len(model_list) > 1:
    print("find multiple models in {}".format(model_path))
    raise ValueError
if len(model_list) < 1:
    print("find no model in {}".format(model_path))
    raise ValueError

model_path = os.path.join(model_path, model_list[0])
print(model_path)
save_dir = "./experiments/results"
data_generator = test_generator()

model = load_model(model_path, custom_objects={"tf": tf, "SGDAccumulate": SGDAccumulate, "score_loss": score_loss,
                                               "siam_loss": siam_loss, "locate_loss": locate_loss,
                                               "score_metric": score_metric, "locate_metric": locate_metric})
count = 0
try:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "predict.txt"), "w") as f:

        while True:
            img1_name, img2_name, img_list, label_list = next(data_generator)
            output = model.predict(img_list, batch_size=1, verbose=0, steps=None)
            output = output[:3]
            count += 1

            f.write(img1_name)
            f.write(",")
            f.write(img2_name)
            f.write("\n")
            f.write(output)
            print(img1_name, img2_name)

except StopIteration:
    print("{} img have been predicted".format(count))
    print("Done")

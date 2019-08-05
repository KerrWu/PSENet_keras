import os
import csv
import numpy as np
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


def result_writer(f, img_name, predict_list, label_list):
    row = [img2_name]
    row.extend(predict_list)
    row.extend(label_list)
    f.writerow(row)


try:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "predict.csv"), "w", newline="") as f:
        csv_writer = csv.writer(f)
        head = ["name", "area", "erythema", "scale", "induration", "pasi", "area_label", "erythema_label",
                "scale_label", "induration_label", "pasi_label"]

        csv_writer.writerow(head)

        while True:
            img1_name, img2_name, img_list, label_list = next(data_generator)
            output = model.predict(img_list, batch_size=1, verbose=0, steps=None)

            img1_result = [elem for elem in output[0].flatten()]
            label1_list = [elem for elem in label_list[0]]

            img2_result = [elem for elem in output[1].flatten()]
            label2_list = [elem for elem in label_list[1]]

            siam_result = [elem for elem in output[2].flatten()]
            label_siam_list = [elem for elem in label_list[2]]

            count += 1

            result_writer(csv_writer, img1_name, img1_result, label1_list)
            result_writer(csv_writer, img2_name, img2_result, label2_list)
            result_writer(csv_writer, img1_name+","+img2_name, siam_result, label_siam_list)

            print(img1_name, img2_name)

except StopIteration:
    print("{} img have been predicted".format(count))
    print("Done")

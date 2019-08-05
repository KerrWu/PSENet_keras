import os
from keras.models import load_model
from data_generator import test_generator

os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_path = "./experiments/checkpoints"

model_list = [elem for elem in os.listdir(model_path) if elem.endswith("h5")]
if len(model_list)>1:
    print("find multiple models in {}".format(model_path))
    raise ValueError
if len(model_list)<1:
    print("find no model in {}".format(model_path))
    raise ValueError

model_path = os.path.join(model_path, model_list[0])
print(model_path)
save_dir = "./experiments/results"
data_generator = test_generator()

model = load_model(model_path)
count = 0
try:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "predict.txt"), "w") as f:

        while True:
            img1_name, img2_name, img_list, label_list = next(data_generator)
            output = model.predict(img_list, batch_size=1, verbose=0, steps=None)
            output = output[:3]
            count+=1

            f.write(img1_name+",")
            f.write(output+"\n")

except StopIteration:
    print("{} img have been predicted".format(count))
    print("Done")





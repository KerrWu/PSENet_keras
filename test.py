import os
from keras.models import load_model
from data_generator import test_generator

model_path = ""
save_dir = ""
data_generator = test_generator()

model = load_model(model_path)
count = 0
try:
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





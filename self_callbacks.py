from keras import callbacks
import os
import numpy as np


class MyEarlyStop(callbacks.Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_mae = float("inf")
        self.best_area = float("inf")
        self.best_ery = float("inf")
        self.best_sca = float("inf")
        self.best_ind = float("inf")
        self.best_pasi = float("inf")


    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        scoreA = logs.get('val_scoreA_score_metric')
        scoreB = logs.get("val_scoreB_score_metric")
        siam = logs.get('val_scoreSiam_score_metric')

        mae = np.mean((scoreA + scoreB + siam)) / 3.0
        mae_area = (scoreA[0] + scoreB[0] + siam[0]) / 3.0
        mae_ery = (scoreA[1] + scoreB[1] + siam[1]) / 3.0
        mae_sca = (scoreA[2] + scoreB[2] + siam[2]) / 3.0
        mae_ind = (scoreA[3] + scoreB[3] + siam[3]) / 3.0
        mae_pasi = (scoreA[4] + scoreB[4] + siam[4]) / 3.0

        global EPOCH
        EPOCH = epoch

        if mae < self.best_mae:
            print("\nValidation average mae decreased from {} to {}, saving model".format(self.best_mae, mae))
            print("\nsave model to " + os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae=" + str(round(mae, 4)) + '.h5'))
            filelist = os.listdir(self.path)
            h5list = [elem for elem in filelist if os.path.splitext(elem)[-1] == '.h5']
            for elem in h5list:
                if "mae" in elem and "mae_" not in elem:
                    os.remove(os.path.join(self.path, elem))
            self.model.save(os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae=" + str(round(mae, 4)) + '.h5'),
                            overwrite=True)
            self.best_mae = mae

        if mae_area < self.best_area:
            print("\nValidation mae area decreased from {} to {}, saving model".format(self.best_area, mae_area))
            print("\nsave model to " + os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_area=" + str(round(mae_area, 4)) + '.h5'))
            filelist = os.listdir(self.path)
            h5list = [elem for elem in filelist if os.path.splitext(elem)[-1] == '.h5']
            for elem in h5list:
                if "mae_area" in elem:
                    os.remove(os.path.join(self.path, elem))
            self.model.save(os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_area=" + str(round(mae_area, 4)) + '.h5'),
                            overwrite=True)
            self.best_area = mae_area

        if mae_ery < self.best_ery:
            print("\nValidation mae ery decreased from {} to {}, saving model".format(self.best_ery, mae_ery))
            print("\nsave model to " + os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_ery=" + str(round(mae_ery, 4)) + '.h5'))
            filelist = os.listdir(self.path)
            h5list = [elem for elem in filelist if os.path.splitext(elem)[-1] == '.h5']
            for elem in h5list:
                if "mae_ery" in elem:
                    os.remove(os.path.join(self.path, elem))
            self.model.save(os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_ery=" + str(round(mae_ery, 4)) + '.h5'),
                            overwrite=True)
            self.best_ery = mae_ery

        if mae_sca < self.best_sca:
            print("\nValidation mae sca decreased from {} to {}, saving model".format(self.best_sca, mae_sca))
            print("\nsave model to " + os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_sca=" + str(round(mae_sca, 4)) + '.h5'))
            filelist = os.listdir(self.path)
            h5list = [elem for elem in filelist if os.path.splitext(elem)[-1] == '.h5']
            for elem in h5list:
                if "mae_sca" in elem:
                    os.remove(os.path.join(self.path, elem))
            self.model.save(os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_sca=" + str(round(mae_sca, 4)) + '.h5'),
                            overwrite=True)
            self.best_sca = mae_sca

        if mae_ind < self.best_ind:
            print("\nValidation mae ind decreased from {} to {}, saving model".format(self.best_ind, mae_ind))
            print("\nsave model to " + os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_ind=" + str(round(mae_ind, 4)) + '.h5'))
            filelist = os.listdir(self.path)
            h5list = [elem for elem in filelist if os.path.splitext(elem)[-1] == '.h5']
            for elem in h5list:
                if "mae_ind" in elem:
                    os.remove(os.path.join(self.path, elem))
            self.model.save(os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_ind=" + str(round(mae_ind, 4)) + '.h5'),
                            overwrite=True)
            self.best_ind = mae_ind

        if mae_pasi < self.best_pasi:
            print("\nValidation mae pasi decreased from {} to {}, saving model".format(self.best_pasi, mae_pasi))
            print("\nsave model to " + os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_pasi=" + str(round(mae_pasi, 4)) + '.h5'))
            filelist = os.listdir(self.path)
            h5list = [elem for elem in filelist if os.path.splitext(elem)[-1] == '.h5']
            for elem in h5list:
                if "mae_pasi" in elem:
                    os.remove(os.path.join(self.path, elem))
            self.model.save(os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae_pasi=" + str(round(mae_pasi, 4)) + '.h5'),
                            overwrite=True)
            self.best_pasi = mae_pasi

        else:
            print("\nValidation have not improvement, current average is {}, best average is {}".format(mae, self.best_mae))
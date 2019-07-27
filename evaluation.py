import os
import sys
from datetime import datetime
from settings import *
from PIL import Image
import numpy as np
from keras.models import model_from_json, load_model
import json
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
np.set_printoptions(threshold=sys.maxsize)


def load(filename):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, "Model_2.h5")
    print(json_file)
    mod = load_model("D:\\Starcraft 2 AI\\Model\\CNN_mode_luke.json")
    print(weight_file)

    return mod


def load_json(filename, weightname):
    json_file = os.path.join(WEIGHTS_DIR, filename)
    weight_file = os.path.join(WEIGHTS_DIR, weightname)
    f = open(json_file, "r")
    mod = model_from_json(f.read())
    print("Model Loaded")
    mod.load_weights(weight_file)
    print("Weights Loaded")
    return mod


def plot_history(hst):
    plt.title("Loss / Mean Squared Error")
    plt.plot(hst.history["loss"], label="train")
    plt.plot(hst.history["val_loss"], label="test")
    plt.legend()
    plt.show()


def load_history():
    history_file = os.path.join(WEIGHTS_DIR, "history_model_10.json")
    return json.load(open(history_file, 'r'))


def plot_history1(his, model_id):
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    acc_file = os.path.join(PLOT_DIR, 'history_plot_acc_'+str(model_id)+'.png')
    loss_file = os.path.join(PLOT_DIR, 'history_plot_loss_'+str(model_id)+'.png')

    history = his.history

    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model '+str(model_id)+' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_file)

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model '+str(model_id)+' Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss_file)


def get_frames(map_name, replay, range_x, range_y):
    proj_dir = FRAMES_DIR + map_name
    frames = []
    for i in range(range_x, range_y+1):
        frame = map_name + "_" + str(replay) + "_frame_" + str(i) + ".png"
        # print(frame)
        im = Image.open(proj_dir + "\\" + frame)
        frame = np.array(im, dtype=np.uint8)
        frames.append(frame)

    frames = np.array(frames)

    return frames



def future_frames_test(id, map, replay, range_x, range_y, future):
    model = load_json("CNN_model_" + str(id) + ".json", "weights_" + str(id) + ".h5")

    # frames = "D:\\Starcraft 2 AI\\Input Frames\\Abyssal_Reef_0_frame_0.png"
    # im = Image.open(frames)
    frames = get_frames(map, replay, range_x, range_y)
    # np_im = np.array(im, dtype=np.int32)
    # np_arr = []
    # np_arr.append(np_im)
    # np.savez("to_predict.npz", x=np_arr)
    # np_arr_2 = np.load("to_predict.npz")
    # np_arr_2 = np_arr_2["x"]
    prediction = frames
    for i in range(1, future):
        prediction = model.predict(prediction)
        p = prediction[0]
        save_prediction(p, id, map, replay, i, i)
        # prediction = prediction[0]  # np.resize(out, (128, 128, 3))


    # save_prediction(prediction, id, map, replay, range_x, range_y)
    # prediction = (prediction).astype(np.uint8)
    # img = Image.fromarray(prediction)
    # img.save("prediction_01a.png")

def single_test(model_id, map_name, replay, lower_bound=0, upper_bound=0):
    model = load_json("CNN_model_" + str(model_id) + ".json", "weights_" + str(model_id) + ".h5")

    frames = get_frames(map_name, replay, lower_bound, upper_bound)

    prediction = model.predict(frames)
    prediction = prediction[0]

    save_prediction(prediction, model_id, map_name, replay, lower_bound, upper_bound)


def callback_predict(model, model_id, epoch_num):
    replay = 141
    lower_bound = 500
    upper_bound = 505
    map_name = "Acid_Plant"
    frames = get_frames(map_name, replay, lower_bound, upper_bound)

    prediction = model.predict(frames)
    prediction = prediction[0]

    save_prediction(prediction, model_id, map_name, replay, lower_bound=lower_bound, upper_bound=upper_bound,
                    epoch_num=epoch_num)


def image_metrics(y, y_hat, show_plot=True, save_plot=False, filename=None):
    if not os.path.exists(METRICS_DIR):
        os.mkdir(METRICS_DIR)

    pred_img = Image.open(y_hat)
    expected_img = Image.open(y)

    pred_img = img_as_float(pred_img)
    expected_img = img_as_float(expected_img)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
    ax = axes.ravel()

    mse_base = mse(expected_img, expected_img)
    ssim_base = ssim(expected_img, expected_img, data_range=expected_img.max() - expected_img.min(), multichannel=True)
    psnr_base = 0  # No way to calculate it as you would have to divide by 0 in the process.

    mse_pred = mse(pred_img, expected_img)
    ssim_pred = ssim(pred_img, expected_img,
                     data_range=pred_img.max() - pred_img.min(), multichannel=True)
    psnr_pred = psnr(pred_img, expected_img)

    label = 'MSE: {:.2f}, SSIM: {:.2f}, PSNR: {:.2f}dB'

    ax[0].imshow(pred_img, vmin=0, vmax=1)
    ax[0].set_xlabel(label.format(mse_base, ssim_base, psnr_base)[:-6]+"infinity")
    ax[0].set_title('Ground Truth')

    ax[1].imshow(expected_img, vmin=0, vmax=1)
    ax[1].set_xlabel(label.format(mse_pred, ssim_pred, psnr_pred))
    ax[1].set_title('Predicted Output')

    lum_img = get_pixel_error(pred_img, expected_img)
    cb = ax[2].imshow(lum_img, vmin=0, vmax=1, cmap='jet')
    ax[2].set_title('Difference Heat Map')
    divider = make_axes_locatable(ax[2])

    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(cb, cax=cax, ax=ax[2])
    plt.tight_layout()

    if save_plot:
        if filename is not None:
            if filename[:-4] != ".png":
                filename = filename + ".png"
            save_file = os.path.join(METRICS_DIR, filename)

            plt.savefig(save_file)
        else:
            save_file = os.path.join(METRICS_DIR, "metrics_plot_"+datetime.today().strftime('%Y-%m-%d %H_%M_%S')+".png")
            plt.savefig(save_file)
    if show_plot:
        plt.show()

    return mse_pred, ssim_pred, psnr_pred


def get_pixel_error(img1, img2):

    # Calculate the absolute difference on each channel separately
    error_r = np.fabs(np.subtract(img2[:, :, 0], img1[:, :, 0]))
    error_g = np.fabs(np.subtract(img2[:, :, 1], img1[:, :, 1]))
    error_b = np.fabs(np.subtract(img2[:, :, 2], img1[:, :, 2]))

    # Calculate the maximum error for each pixel
    lum_img = np.maximum(np.maximum(error_r, error_g), error_b)

    return lum_img


def save_prediction(prediction, model_id, map_name, replay, lower_bound=0, upper_bound=0, epoch_num=None):
    pred_dir = os.path.join(PREDICTION_DIR, "Model_" + str(model_id))
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    prediction = prediction.astype(np.uint8)
    img = Image.fromarray(prediction)
    if epoch_num is None:
        img.save(pred_dir + "\\prediction_" + map_name + "_" + str(replay) + "_" + str(lower_bound) + "-" +
                 str(upper_bound) + ".png")
    else:
        img.save(pred_dir + "\\prediction_" + map_name + "_" + str(replay) + "_" + str(lower_bound) + "-" +
                 str(upper_bound) + "_epoch_" + str(epoch_num) + ".png")


def mse(x, y):
    return np.linalg.norm(x - y)


if __name__ == "__main__":
    # model = load_json("CNN_model_01.json", "weights_01.h5")
    single_test(15, "Acid_Plant", 141, 1500, 1500)
    # single_test(10, "Acid_Plant", 141, 1496, 1498)
    image_metrics("output.png", "prediction.png", save_plot=True)
    print("Evaluation Complete")
    # hst = load_history()
    # plot_history(hst)

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
    plt.clf()
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
    for i in range(range_x, range_y):
        frame = map_name + "_" + str(replay) + "_frame_" + str(i) + ".png"
        # print(frame)
        im = Image.open(proj_dir + "\\" + frame)
        frame = np.array(im, dtype=np.uint8)
        frames.append(frame)
    pred = []
    pred.append(frames)
    pred = np.array(pred, dtype=np.uint8)

    return pred, frames


def test_RNN(id, map, replay, lower_bound, upper_bound):
    model = load_json("CNN_model_" + str(id) + ".json", "weights_" + str(id) + ".h5")

    frames = get_frames(map, replay, lower_bound, upper_bound)
    prediction = model.predict(frames)
    p = prediction[0]
    save_prediction(p, id, map, replay, lower_bound, upper_bound)


def future_frames_CNN(map, replay, range_x, range_y, future):
    model = load_json("CNN_model_8_bp.json", "weights_8_ct.h5")

    frames = get_frames(map, replay, range_x, range_y)
    prediction = frames
    for i in range(0, future):
        prediction = model.predict(prediction)
        p = prediction[0]
        img = Image.fromarray(p.astype('uint8'))
        img.save("D:\\Starcraft 2 AI\\Results\\Buildings\\Prediction_" + str(i) + ".png")


def future_frames_RNN(map, replay, range_x, range_y, future):
    model = load_json("CNN_model_9_CT7.json", "weights_9_CT7.h5")

    frames, f = get_frames(map, replay, range_x, range_y)
    prediction = frames
    for i in range(0, future):
        pred = model.predict(prediction)
        p = pred[0]
        img = Image.fromarray(p.astype('uint8'))
        img.save("D:\\Starcraft 2 AI\\Results\\Future_Frames\\Prediction_" + str(i) + ".png")
        f.pop()
        f.append(p)
        prediction = []
        prediction.append(f)
        prediction = np.array(prediction, dtype=np.uint8)


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
                    epoch_num=epoch_num, y_true="input.png")


def image_metrics(y_true, y_pred, x=None, show_plot=True, save_plot=True, filename=None):
    if not os.path.exists(METRICS_DIR):
        os.mkdir(METRICS_DIR)

    expected_img = Image.open(y_true)
    expected_img = img_as_float(expected_img)

    pred_img = Image.open(y_pred)
    pred_img = img_as_float(pred_img)
    if x is not None:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 8))
        input_img = Image.open(x)
        input_img = img_as_float(input_img)
    else:
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

    if x is not None:
        mse_input = mse(input_img, expected_img)
        ssim_input = ssim(input_img, expected_img,
                          data_range=pred_img.max() - pred_img.min(), multichannel=True)
        psnr_input = psnr(input_img, expected_img)

        ax[0].imshow(input_img, vmin=0, vmax=1)
        ax[0].set_xlabel(label.format(mse_base, ssim_base, psnr_base)[:-6] + "infinity")
        ax[0].set_title('Input Image')

        ax[1].imshow(expected_img, vmin=0, vmax=1)
        ax[1].set_xlabel(label.format(mse_input, ssim_input, psnr_input))
        ax[1].set_title('Ground Truth')

        lum_input_img = get_pixel_error(input_img, expected_img)
        cb = ax[2].imshow(lum_input_img, vmin=0, vmax=1, cmap='jet')
        ax[2].set_title('Input/Output Difference Heat Map')
        divider = make_axes_locatable(ax[2])

        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(cb, cax=cax, ax=ax[2])
        ax[3].imshow(expected_img, vmin=0, vmax=1)
        ax[3].set_xlabel(label.format(mse_base, ssim_base, psnr_base)[:-6] + "infinity")
        ax[3].set_title('Ground Truth')

        ax[4].imshow(pred_img, vmin=0, vmax=1)
        ax[4].set_xlabel(label.format(mse_pred, ssim_pred, psnr_pred))
        ax[4].set_title('Predicted Output')

        lum_img = get_pixel_error(pred_img, expected_img)
        cb = ax[5].imshow(lum_img, vmin=0, vmax=1, cmap='jet')
        ax[5].set_title('Difference Heat Map')
        divider = make_axes_locatable(ax[5])

        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(cb, cax=cax, ax=ax[2])
    else:
        ax[0].imshow(expected_img, vmin=0, vmax=1)
        ax[0].set_xlabel(label.format(mse_base, ssim_base, psnr_base)[:-6]+"infinity")
        ax[0].set_title('Ground Truth')

        ax[1].imshow(pred_img, vmin=0, vmax=1)
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
            save_file = os.path.join("D:\\Starcraft 2 AI\\Results\\Future_Frames\\", filename)

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


def save_prediction(prediction, model_id, map_name, replay, lower_bound=0, upper_bound=0, epoch_num=None, y_true=None):
    pred_dir = os.path.join(PREDICTION_DIR, "Model_" + str(model_id))
    if not os.path.exists(PREDICTION_DIR):
        os.mkdir(PREDICTION_DIR)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    prediction = prediction.astype(np.uint8)
    img = Image.fromarray(prediction)
    if epoch_num is None:
        save_path = os.path.join(pred_dir, "prediction_" + map_name + "_" + str(replay) + "_" + str(lower_bound)
                                 + "-" + str(upper_bound) + ".png")

        img.save(save_path)
    else:
        save_path = os.path.join(pred_dir, "prediction_" + map_name + "_" + str(replay) + "_" + str(lower_bound)
                                 + "-" + str(upper_bound) + "_epoch_" + str(epoch_num) + ".png")

        img.save(save_path)
        if y_true is not None:
            metric_filename = "Model_" + str(model_id) + "_Epoch_" + epoch_num
            image_metrics(y_true, save_path, show_plot=False, save_plot=True, filename=metric_filename)


def mse(x, y):
    return np.linalg.norm(x - y)


if __name__ == "__main__":
    future_frames_RNN("Catalyst", 108, 225, 228, 1)

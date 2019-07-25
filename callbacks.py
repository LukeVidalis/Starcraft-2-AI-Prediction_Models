from keras.callbacks import ModelCheckpoint
from evaluation import callback_predict


class CallbackPred(ModelCheckpoint):
    def __init__(self, period, model_id):
        self.period = period
        self.epochs_since_last_save = 0
        self.model_id = model_id

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            callback_predict(self.model, model_id=self.model_id, epoch_num=epoch)

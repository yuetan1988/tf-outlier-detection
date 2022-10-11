import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tfod.models.rnn import RNN
from matplotlib.pyplot import figure


class TsModel(object):
    def __init__(self, use_model) -> None:
        self.use_model = use_model
        if use_model.lower() == "rnn":
            self.Model = RNN()

    def __call__(self, x):
        return self.Model(x)

    # def from_checkpoint(self, model_dir=None):
    #     model = self.Model.load_weights(model_dir)
    #     return model


class OdModel(object):
    """Reconstruct model"""

    def __init__(self, model, train_sequence_length) -> None:
        self.model = model
        self.train_sequence_length = train_sequence_length

    def detect(self, x_test, y_test, plot=False):
        y_pred = self.model(x_test)
        y_pred = y_pred.numpy()
        errors = y_pred - y_test

        # mean/cov
        mean = sum(errors) / len(errors)
        cov = 0
        for e in errors:
            cov += np.dot((e - mean).reshape(len(e), 1), (e - mean).reshape(1, len(e)))
        cov /= len(errors)

        m_dist = [0] * self.train_sequence_length
        for e in errors:
            m_dist.append(mahala_distantce(e, mean, cov))

        return m_dist

    def plot(self, sig, det):
        # figure(figsize=(6, 4), dpi=80)
        fig, axes = plt.subplots(nrows=2, figsize=(4, 3))

        axes[0].plot(sig, color="b", label="original data")
        x = np.arange(4200, 4400)
        y1 = [-3] * len(x)
        y2 = [3] * len(x)
        axes[0].fill_between(x, y1, y2, facecolor="g", alpha=0.3)

        axes[1].plot(det, color="r", label="Mahalanobis Distance")
        axes[1].set_ylim(0, 1000)
        y1 = [0] * len(x)
        y2 = [1000] * len(x)
        axes[1].fill_between(x, y1, y2, facecolor="g", alpha=0.3)
        # plt.savefig('./p.png')
        # plt.show()


# calculate Mahalanobis distance
def mahala_distantce(x, mean, cov):
    d = np.dot(x - mean, np.linalg.inv(cov))
    d = np.dot(d, (x - mean).T)
    return d

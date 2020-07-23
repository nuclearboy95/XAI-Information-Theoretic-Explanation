# Information-theoretic attribution definitions
import numpy as np
from collections import defaultdict

eps = 1e-13


__all__ = ['AttrCalculator', 'ITAttr']


def div(x, y):
    sign = np.sign(y)
    margin = sign * eps + (sign == 0) * eps
    return x / (y + margin)


class AttrCalculator:
    def __init__(self, H, W):
        self.counts = defaultdict(lambda: np.zeros((H, W), dtype=np.int32))
        self.scores = defaultdict(lambda: np.zeros((H, W), dtype=np.float32))

    def compute(self, key, attr_f, p_Y_x, p_Y_Xpert, mask, y=None):
        score = attr_f(p_Y_x, p_Y_Xpert, y)
        self.scores[(key, y)][mask] += score
        self.counts[(key, y)][mask] += 1

    def get_result(self, key, y=None):
        return self.scores[(key, y)] / self.counts[(key, y)]


class ITAttr:
    @staticmethod
    def PMI_MC(p_Y_x, p_Y_Xs, y):
        """
        :param np.ndarray p_Y_x: [Y] float32 output prediction of model for original image
        :param np.ndarray p_Y_Xs: [N, Y] float32 output prediction of model for each x_i
        :param int y: class
        :return:
        """
        p_y_xi = p_Y_x[y]  # []
        p_y_Xs = p_Y_Xs[:, y]  # [N]
        p_y = E_p_y_Xi = np.mean(p_y_Xs)
        PMI = np.log(eps + div(p_y_xi, p_y))
        return PMI

    @staticmethod
    def IG_MC(p_Y_x, p_Y_Xs, y=None):
        """
        :param np.ndarray p_Y_x: [Y] float32 output prediction of model for original image
        :param np.ndarray p_Y_Xs: [N, Y] float32 output prediction of model for each x_i
        :param int y: class
        :return:
        """
        # AXIS: [Y, N]
        p_Y_x = p_Y_x[:, np.newaxis]  # [Y, 1]
        p_Y_Xs = np.swapaxes(p_Y_Xs, 0, 1)  # [Y, N]

        if y is not None:
            p_y_Xs = p_Y_Xs[y, np.newaxis, :]  # [1, N]
            p_Y_Xs = np.concatenate([p_y_Xs, 1 - p_y_Xs], axis=0)  # [Y=2, N]

            p_y_x = p_Y_x[y, np.newaxis, :]  # [1, 1]
            p_Y_x = np.concatenate([p_y_x, 1 - p_y_x], axis=0)  # [Y=2, 1]

        p_Y = E_p_Y_Xi = np.mean(p_Y_Xs, axis=1, keepdims=True)  # [Y, 1]
        PMI_Y = np.log(eps + div(p_Y_x, p_Y))  # [Y, 1]
        IG = np.sum(p_Y_x * PMI_Y)
        return IG  # abs(IG * sign)

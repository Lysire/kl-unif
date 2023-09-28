from decimal import *
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import comb

# save figures here
SAVE_PATH = Path.home() / "fyp" / "hessian_plots"

# NOTE: There's no point analysing the behaviour of
# f since the function shapes are wonky, as seen on Desmos

# let the term in the summation for h_ii be f(m, k, p_i)
def f(m, k, p: Decimal):
    front = Decimal(comb(m - 1, k - 1))
    middle = Decimal(np.log2(k)) * (p ** (k - 2)) * ((1 - p) ** (m - k - 2))
    back = k * (k - 1 - 2 * (m - 1) * p) + m * (m - 1) * (p ** 2)

    return front * middle * back


# compute h_ii from p and m (number of samples)
# use Decimal since probability powers can be very small
def h_ii(m, p: Decimal):
    one_iter = lambda k : f(m, k, p)
    value = sum(map(one_iter, range(2, m + 1)))
    return value

# plot h_ii against m for different values of m
# over p in (0, 1), remove p = 0 and p = 1 endpoints since
# p = 0 causes 0 ** 0 invalid operation
# it is problematic for (1 - p) ** (m - k - 2)
def plot_h_ii(m):
    x_axis = np.linspace(0.001, 1, endpoint=False)  # x-axis
    hessians = []  # y-axis

    for p in x_axis:
        p_dec = Decimal(p)
        hessians.append(np.longdouble(h_ii(m, p_dec)))

    hessians = np.array(hessians)

    # clear previous plots
    plt.clf()

    # update plot details
    plt.plot(x_axis, hessians)
    plt.xlabel("p")
    plt.ylabel("h_ii")

    # save plot
    plt_name = f"hessian_plot_{m}.png"
    plt.savefig(SAVE_PATH / plt_name, bbox_inches='tight')


if __name__ == "__main__":
    for m in range(2, 40, 2):
        plot_h_ii(m)

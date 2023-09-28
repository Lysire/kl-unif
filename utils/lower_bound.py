from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# save figures here
SAVE_PATH = Path.home() / "fyp" / "lb_plots"

# generates distribution p of the form
# (1/k, ..., 1/k, ..., 0)
def gen_k_uniform(k, n):
    assert k < n

    res = [0 for _ in range(n)]
    for i in range(k):
        res[i] = 1 / k

    return np.array(res)


# calculate KL-divergence between p and U_n
def kl_with_unif(p):
    n = len(p)
    p = np.array(p, dtype=np.float)
    u_n = np.array([1 / n for _ in range(n)], dtype=np.float)

    res = 0
    for a, b in zip(p, u_n):
        if a == 0:
            continue
        res += a * np.log(a / b)

    return res


# given p, generate p' where p' is the distribution obtained as in the report
# (avg heaviest n // 2 elts, and smallest n // 2 elts)
def gen_prime_dist(p):
    n = len(p)
    middle = n // 2

    p_sorted = -np.sort(-p)  # descending order
    p_prime = np.copy(p_sorted)

    # average heaviest n // 2 elts
    heaviest_avg = sum(p_sorted[:middle]) / middle
    for i in range(middle):
        p_prime[i] = heaviest_avg

    middle_start = middle + 1 if n % 2 == 1 else middle

    # average smallest n // 2 elts
    smallest_avg = sum(p_sorted[middle_start:]) / middle
    for i in range(middle_start, n):
        p_prime[i] = smallest_avg

    return np.array(p_prime)


# given p, calculate lower bound \alpha = D_KL(p' || U_n) / D_KL(p || U_n)
def get_lower_bound(p):
    p_prime = gen_prime_dist(p)

    return kl_with_unif(p_prime) / kl_with_unif(p)


# compare lower bound \alpha across (1, 0, ..., 0)
# to (1/(n-1), ..., 1/(n-1), 0)
def plot_lower_bounds_for_k_unif(n):
    k_vals = np.arange(1, n)  # x-axis
    lower_bounds = []  # y-axis

    for k in range(1, n):
        dist = gen_k_uniform(k, n)
        lower_bounds.append(get_lower_bound(dist))

    lower_bounds = np.array(lower_bounds)

    # clear previous plots
    plt.clf()

    # update plot details
    plt.plot(k_vals, lower_bounds)
    plt.xlabel("k")
    plt.ylabel("lower bound")

    # save plot
    plt_name = f"lower_bounds_{n}.png"
    plt.savefig(SAVE_PATH / plt_name, bbox_inches='tight')


# compare across different n from n >= 3 onwards
def create_plots(start_n = 3, end_n = 20):
    for n in range(start_n, end_n + 1):
        plot_lower_bounds_for_k_unif(n)


if __name__ == "__main__":
    create_plots()

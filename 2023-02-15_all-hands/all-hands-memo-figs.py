import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.special import expit, logit


# Figure 3
def plot_beta_panel(ax, x, p0, ymax, a1, b1, a2, b2, **kwargs):
    ax.plot(x, beta.pdf(x, a1, b1), color="b", label="non-threat")
    ax.plot(x, beta.pdf(x, a2, b2), color="r", label="threat")
    ax.vlines(p0, 0, ymax, color="0.25", linestyle="dashed")

    ax.set_xlim([0, 1])
    ax.set_xticks([0, p0, 1])
    ax.set_xticklabels([0, r"$p_0$", 1])

    ax.set_ylim([0, ymax])
    ax.set_yticks([])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)


fig = plt.figure(figsize=(8, 2.5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

margin = 0.01
x = np.linspace(margin, 1 - margin, 600)

p0 = 0.25
shape = 10
a = shape
b = a * (1 - p0) / p0
ymax = 8

delta1 = 4
delta2 = 150
plot_beta_panel(ax1, x, p0, 7, a, b + delta1, a + delta1, b)
plot_beta_panel(ax2, x, p0, 28, a, b + delta2, a + delta2, b)
ax1.set_ylabel(r"Probability density, $Pr\{P[T|X]\}$")
fig.text(0.5, -0.05, r"Threat probability after observing data, $P[T|X]$", ha="center")
ax2.legend(frameon=False)
ax1.set_title("Lower-quality data")
ax2.set_title("Higher-quality data")
fig.savefig("figure3.pdf", bbox_inches="tight")

# Figure 5
fig = plt.figure(figsize=(8, 2.5))
rng = np.random.default_rng(seed=100)


def p_trajectory(p0, drift, diff, n):
    delta_y = rng.normal(loc=drift, scale=np.sqrt(diff), size=n - 1)
    y0 = logit(p0)
    y = y0 * np.ones(n, dtype=float)
    y[1:] += np.cumsum(delta_y)
    return expit(y)


batch = np.arange(0, 20, 1)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
num_reps = 4
p_thresh = 0.475

taus = [0.1, 1.2]
for ax, tau in zip(fig.axes, taus):
    for i in range(num_reps):
        (l_nt,) = ax.plot(batch, p_trajectory(p0, -tau / 2, tau, len(batch)), color="b")
        (l_t,) = ax.plot(batch, p_trajectory(p0, tau / 2, tau, len(batch)), color="r")
        if i == 0:
            l_t.set_label("threat")
            l_nt.set_label("non-threat")
    ax.hlines(p_thresh, 0, len(batch), linestyle="dotted", color="0.25")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(batch)])
ax2.legend(frameon=False)
ax1.set_yticks([0, p0, p_thresh, 1])
ax1.set_yticklabels([0, r"$p_0$", r"$p_{thresh}$", 1])
ax1.set_ylabel(r"Posterior probability, $P[T|X]$")
ax2.set_yticks([])
ax2.set_yticklabels([])
fig.text(0.5, -0.05, r"Data batch, $i$", ha="center")
ax1.set_title("Lower-quality data")
ax2.set_title("Higher-quality data")
fig.savefig("figure5.pdf", bbox_inches="tight")

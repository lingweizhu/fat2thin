import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def c240718():
    def plot(pths):
        fig, axs = plt.subplots(len(envs), len(datasets),
                                figsize=(4 * len(datasets), 3 * len(envs)))
        axs = np.asarray([axs])
        axs = axs.reshape((len(envs), len(datasets)))
        for i, e in enumerate(envs):
            for j, d in enumerate(datasets):
                ed_pth = fill_in_path(pths, [e, d])
                learning_curve(axs[i][j], ed_pth, 10, colors, styles)
                axs[i][j].set_title("{} {}".format(e, d))
        # plt.legend()
        plt.tight_layout()
        return axs

    colors = {
        "TTT": "C0",
        "Gan": "C1",
        "JS": "C2",
        "Jeffery": "C3",
    }
    styles = {
        "TTT": "-",
        "Gan": "-",
        "JS": "-",
        "Jeffery": "-",
    }
    envs = ["HalfCheetah"]
    datasets = ["medrep"]

    q_list = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',
              '-0.1', '-0.2', '-0.3', '-0.4', '-0.5', '-0.6', '-0.7', '-0.8', '-0.9']
    pth_base = "../data/output/test_v0/test_TTT/q_0.0_{}/fdiv_4_gan/{}/{}/TTT/HTqGaussian/"
    pths = {}
    for q in q_list:
        pths[q] = pth_base.format(q, "{}", "{}")
    axs = plot(pths)
    plt.savefig("img/q_gan_terms4.png", dpi=300)

    pth_base = "../data/output/test_v0/test_TTT/q_0.0_{}/fdiv_4_jeffrey/{}/{}/TTT/HTqGaussian/"
    pths = {}
    for q in q_list:
        pths[q] = pth_base.format(q, "{}", "{}")
    axs = plot(pths)
    plt.savefig("img/q_jeffrey_terms4.png", dpi=300)

    pth_base = "../data/output/test_v0/test_TTT/q_0.0_{}/fdiv_4_jensen_shannon/{}/{}/TTT/HTqGaussian/"
    pths = {}
    for q in q_list:
        pths[q] = pth_base.format(q, "{}", "{}")
    axs = plot(pths)
    plt.savefig("img/q_jensen_shannon_terms4.png", dpi=300)


if __name__ == "__main__":
    c240718() # TTT

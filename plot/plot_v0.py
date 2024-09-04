import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def c240604():
    pths = {
        "FTT-AWAC": "../data/output/test_v0/{}/{}/FTT_AWAC/qGaussian/",
    }
    colors = {
        "FTT-AWAC": "C0",
    }
    styles = {
        "FTT-AWAC": "-",
    }
    envs = ["Hopper"]
    datasets = ["expert"]
    fig, axs = plt.subplots(len(datasets), len(envs), figsize=(5*len(envs), 4*len(datasets)))
    if len(datasets) == 1:
        axs = [axs]
    if len(datasets) == 1:
        axs = [axs]
    for ie, e in enumerate(envs):
        for id_, d in enumerate(datasets):
            ed_pth = fill_in_path(pths, [e, d])
            learning_curve_sweep(axs[id_][ie], ed_pth, smoothing=10)
            axs[id_][ie].set_title("{}-{}".format(e, d))
    plt.show()

def c240615():
    pths = {
        "TTT-0.5-0.0": "../data/output/test_v0/test_TTT/q_0.5_0.0/fdiv_{}_{}/Ant/expert/TTT/Student/",
        "TTT-0.0-0.0": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_{}_{}/Ant/expert/TTT/Student/",
    }
    colors = {
        "TTT-0.5-0.0": "C0",
        "TTT-0.0-0.0": "C1",
    }
    styles = {
        "TTT-0.5-0.0": "-",
        "TTT-0.0-0.0": "-",
    }
    envs = ["Ant"]
    datasets = ["expert"]
    distance = ['gan', 'jensen_shannon', 'jeffrey']
    terms = [3,4,5,6]
    fig, axs = plt.subplots(len(terms), len(distance), figsize=(5*len(distance), 4*len(terms)))
    for i, t in enumerate(terms):
        for j, d in enumerate(distance):
            if not (t==6 and d=='jeffrey'):
                ed_pth = fill_in_path(pths, [t, d])
                learning_curve(axs[i][j], ed_pth, 10, colors, styles)
                # learning_curve_sweep(axs[i][j], ed_pth, smoothing=10)
                axs[i][j].set_title("{} {}".format(d, t))
    plt.tight_layout()
    plt.show()

def c240615():
    def plot():
        fig, axs = plt.subplots(len(envs), len(datasets),
                                figsize=(4 * len(datasets), 3 * len(envs)))
        axs = axs.reshape((len(envs), len(datasets)))
        for i, e in enumerate(envs):
            for j, d in enumerate(datasets):
                ed_pth = fill_in_path(pths, [e, d])
                learning_curve(axs[i][j], ed_pth, 10, colors, styles)
                # learning_curve_sweep(axs[i][j], ed_pth, smoothing=10)
                axs[i][j].set_title("{} {}".format(e, d))
        plt.legend()
        plt.tight_layout()
        return axs

    colors = {
        "TTT": "C0",
    }
    styles = {
        "TTT": "-",
    }
    envs = ["HalfCheetah"]
    datasets = ["medexp", "medrep"]
    distance = ['gan', 'jensen_shannon', 'jeffrey']

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_2_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_2_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_2_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    # plt.show()
    plt.savefig("img/terms2.png", dpi=300)

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/terms4.png", dpi=300)

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_6_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_6_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_6_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/terms6.png", dpi=300)

def c240615_1():
    colors = {
    }
    styles = {
    }
    envs = ["Hopper"]
    datasets = ["medexp"]

    pths = {
        "sample from actor": "../data/output/test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_CopyMu-Pi/{}/{}/FTT/qGaussian/",
        "sample from proposal": "../data/output/test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_CopyMu-Proposal/{}/{}/FTT/qGaussian/",
    }
    fig, axs = plt.subplots(len(envs), len(datasets),
                            figsize=(4 * len(datasets), 3 * len(envs)))
    axs = [[axs]]
    for i, e in enumerate(envs):
        for j, d in enumerate(datasets):
            ed_pth = fill_in_path(pths, [e, d])
            learning_curve(axs[i][j], ed_pth, 10, colors, styles)
            axs[i][j].set_title("{} {}".format(e, d))
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/ftt.png", dpi=300)

def c240620():
    def plot():
        fig, axs = plt.subplots(len(envs), len(datasets),
                                figsize=(4 * len(datasets), 3 * len(envs)))
        axs = axs.reshape((len(envs), len(datasets)))
        for i, e in enumerate(envs):
            for j, d in enumerate(datasets):
                ed_pth = fill_in_path(pths, [e, d])
                learning_curve(axs[i][j], ed_pth, 10, colors, styles)
                # learning_curve_sweep(axs[i][j], ed_pth, smoothing=10)
                axs[i][j].set_title("{} {}".format(e, d))
        plt.legend()
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
    datasets = ["medexp", "medrep"]
    distance = ['gan', 'jensen_shannon', 'jeffrey']

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_2_gan/{}/{}/TTT/Student/",
        # "JS": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_2_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_2_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    # plt.show()
    plt.savefig("img/v2_terms2.png", dpi=300)

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_4_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_4_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_4_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/v2_terms4.png", dpi=300)

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_6_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_6_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT_v2/q_0.0_0.0/fdiv_6_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/v2_terms6.png", dpi=300)


def c240622():
    def plot():
        fig, axs = plt.subplots(len(envs), len(datasets),
                                figsize=(4 * len(datasets), 3 * len(envs)))
        axs = np.asarray([axs])
        axs = axs.reshape((len(envs), len(datasets)))
        for i, e in enumerate(envs):
            for j, d in enumerate(datasets):
                ed_pth = fill_in_path(pths, [e, d])
                learning_curve(axs[i][j], ed_pth, 10, colors, styles)
                # learning_curve_sweep(axs[i][j], ed_pth, smoothing=10)
                axs[i][j].set_title("{} {}".format(e, d))
        plt.legend()
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
    distance = ['gan', 'jensen_shannon', 'jeffrey']

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT/q_1.0_0.0/fdiv_4_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT/q_1.0_0.0/fdiv_4_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT/q_1.0_0.0/fdiv_4_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/q_1_0_terms4.png", dpi=300)

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT/q_0.5_0.5/fdiv_4_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT/q_0.5_0.5/fdiv_4_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT/q_0.5_0.5/fdiv_4_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/q_0.5_0.5_terms4.png", dpi=300)

def c240623():
    def plot():
        fig, axs = plt.subplots(len(envs), len(datasets),
                                figsize=(4 * len(datasets), 3 * len(envs)))
        axs = np.asarray([axs])
        axs = axs.reshape((len(envs), len(datasets)))
        for i, e in enumerate(envs):
            for j, d in enumerate(datasets):
                ed_pth = fill_in_path(pths, [e, d])
                learning_curve(axs[i][j], ed_pth, 10, colors, styles)
                # learning_curve_sweep(axs[i][j], ed_pth, smoothing=10)
                axs[i][j].set_title("{} {}".format(e, d))
        plt.legend()
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
    distance = ['gan', 'jensen_shannon', 'jeffrey']

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_3_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_3_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_3_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/v3_terms3.png", dpi=300)

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_4_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_4_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_4_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/v3_terms4.png", dpi=300)

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_5_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_5_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT_v3/q_0.0_0.0/fdiv_5_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/v3_terms5.png", dpi=300)

def c240625():
    def plot():
        fig, axs = plt.subplots(len(envs), len(datasets),
                                figsize=(4 * len(datasets), 3 * len(envs)))
        axs = np.asarray([axs])
        axs = axs.reshape((len(envs), len(datasets)))
        for i, e in enumerate(envs):
            for j, d in enumerate(datasets):
                ed_pth = fill_in_path(pths, [e, d])
                learning_curve(axs[i][j], ed_pth, 10, colors, styles)
                # learning_curve_sweep(axs[i][j], ed_pth, smoothing=10)
                axs[i][j].set_title("{} {}".format(e, d))
        plt.legend()
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
    distance = ['gan', 'jensen_shannon', 'jeffrey']

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT/q_0.5_0.0/fdiv_4_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT/q_0.5_0.0/fdiv_4_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT/q_0.5_0.0/fdiv_4_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/q_0.5_0.0_terms4.png", dpi=300)

    pths = {
        "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_gan/{}/{}/TTT/Student/",
        "JS": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_jensen_shannon/{}/{}/TTT/Student/",
        "Jeffery": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_jeffrey/{}/{}/TTT/Student/",
    }
    axs = plot()
    plt.savefig("img/q_0.0_0.0_terms4.png", dpi=300)

def c240702():
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
        plt.legend()
        plt.tight_layout()
        return axs

    def sweep(pths, fixed_param={"q_lr_prob": 1./3.}, x_axis=["pi_lr", [1e-3, 3e-4, 1e-4]], y_axis=["tau", [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]]):
        fig, axs = plt.subplots(len(envs), len(datasets),
                                figsize=(4 * len(datasets), 3 * len(envs)))
        axs = np.asarray([axs])
        axs = axs.reshape((len(envs), len(datasets)))
        for i, e in enumerate(envs):
            for j, d in enumerate(datasets):
                ed_pth = fill_in_path(pths, [e, d])
                ed_pth = list(ed_pth.values())[0]
                # learning_curve_sweep(axs[i][j], ed_pth, smoothing=10, key_params=["pi_lr", "q_lr_prob", "tau"])
                # filter_target_parameter(axs[i][j], ed_pth, smoothing=10, key_param_values={"pi_lr": 0.0003,
                #                                                                      "q_lr_prob": 1.0})
                draw_heatmap(axs[i][j], ed_pth, smoothing=10, fixed_param=fixed_param, x_axis=x_axis, y_axis=y_axis)
                axs[i][j].set_title("{} {}".format(e, d))
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

    pths = {
        # "TTT": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_1/{}/{}/TTT/Student/",
        "Gan": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_gan/{}/{}/TTT/Student/",
        # "Jeffery": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_jeffrey/{}/{}/TTT/Student/",
    }
    # axs = plot(pths)
    pths = {
        "Gan": "../data/output/test_v0/test_TTT/q_0.0_0.0/fdiv_4_gan/{}/{}/TTT/Student/",
    }
    axs = sweep(pths, fixed_param={"q_lr_prob": 1./3.})
    plt.savefig("img/q_0.0_0.0_terms4_largesweep_qlrmulti0.3.png", dpi=300)
    axs = sweep(pths, fixed_param={"q_lr_prob": 1.})
    plt.savefig("img/q_0.0_0.0_terms4_largesweep_qlrmulti1.png", dpi=300)
    axs = sweep(pths, fixed_param={"q_lr_prob": 3.})
    plt.savefig("img/q_0.0_0.0_terms4_largesweep_qlrmulti3.png", dpi=300)

def c240711():
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
        plt.legend()
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
    pth_base = "../data/output/test_v0/test_TTT/q_0.0_{}/fdiv_4_gan/{}/{}/TTT/Student/"
    pths = {}
    for q in q_list:
        pths[q] = pth_base.format(q, "{}", "{}")
    axs = plot(pths)
    plt.savefig("img/q_gan_terms4.png", dpi=300)

    pth_base = "../data/output/test_v0/test_TTT/q_0.0_{}/fdiv_4_jeffrey/{}/{}/TTT/Student/"
    pths = {}
    for q in q_list:
        pths[q] = pth_base.format(q, "{}", "{}")
    axs = plot(pths)
    plt.savefig("img/q_jeffrey_terms4.png", dpi=300)

    pth_base = "../data/output/test_v0/test_TTT/q_0.0_{}/fdiv_4_jensen_shannon/{}/{}/TTT/Student/"
    pths = {}
    for q in q_list:
        pths[q] = pth_base.format(q, "{}", "{}")
    axs = plot(pths)
    plt.savefig("img/q_jensen_shannon_terms4.png", dpi=300)


def c240716():
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
        plt.legend()
        plt.tight_layout()
        return axs

    colors = {
        "FTT-AWAC": "C0",
    }
    styles = {
        "FTT-AWAC": "-",
    }
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medium", "medrep"]
    pths = {
        "FTT": "../data/output/test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_CopyMu-Pi/{}/{}/FTT/qGaussian/",
    }
    axs = plot(pths)
    plt.savefig("img/ftt.png", dpi=300)

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


def c240730():
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
        plt.legend()
        plt.tight_layout()
        return axs

    colors = {
    }
    styles = {
    }
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medium", "medrep"]
    pths = {
        "Copy": "../data/output/test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_CopyMu-Pi/{}/{}/FTT/qGaussian/",
        "KL": "../data/output/test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "SPOT": "../data/output/test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/{}/{}/FTT/qGaussian/",
    }
    axs = plot(pths)
    plt.savefig("img/ftt.png", dpi=300)

def c240815():
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
        plt.legend()
        plt.tight_layout()
        return axs

    colors = {
    }
    styles = {
    }
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medium", "medrep"]
    pths = {
        "FTT": "../data/output/test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "IQL": "../data/baseline_data/output/test_v1/{}/{}/IQL/HTqGaussian/",
        "InAC": "../data/baseline_data/output/test_v1/{}/{}/InAC/HTqGaussian/",
        "TAWAC": "../data/baseline_data/output/test_v1/{}/{}/TAWAC/HTqGaussian/",
    }
    axs = plot(pths)
    plt.savefig("img/ftt.png", dpi=300)


if __name__ == "__main__":
    # extract_best_param()
    # find_suboptimal_setting()
    # check_param_consistancy()
    # c240718() # TTT
    # c240730() # FTT
    c240815() # FTT


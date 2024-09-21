import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        # "IQL-HTqG": "../data/baseline_data/output/test_v1/{}/{}/IQL/HTqGaussian/",
        "IQL-SG": "../data/baseline_data/output/test_v1/{}/{}/IQL/SGaussian/",
        # "InAC-HTqG": "../data/baseline_data/output/test_v1/{}/{}/InAC/HTqGaussian/",
        "InAC-SG": "../data/baseline_data/output/test_v1/{}/{}/InAC/SGaussian/",
        # "TAWAC-HTqG": "../data/baseline_data/output/test_v1/{}/{}/TAWAC/HTqGaussian/",
        "TAWAC-SG": "../data/baseline_data/output/test_v1/{}/{}/TAWAC/SGaussian/",
    }
    axs = plot(pths)
    plt.savefig("img/ftt.png", dpi=300)


def c240826():
    # plot projection baseline
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
    envs = ["HalfCheetah"]
    datasets = ["medexp", "medium"]
    pths = {
        "FTT": "../data/output/test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "Baseline1": "../data/output/test_v0/naive_projection_baselines/{}/{}/TAWACqG/qGaussian/",
        "Baseline2": "../data/output/test_v0/naive_projection_baselines/{}/{}/TAWACqGC/qGaussian/",
    }
    axs = plot(pths)
    plt.savefig("img/projection_baselines.png", dpi=300)

def c240828():
    # plot projection baseline
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
    envs = ["SimEnv3"]
    datasets = ["random"]
    pths = {
        "FTT-HT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "FTT": "../data/output/test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "IQL": "../data/output/test_v1/baseline/{}/{}/IQL/SGaussian/",
        "TAWAC": "../data/output/test_v1/baseline/{}/{}/TAWAC/SGaussian/",
    }
    axs = plot(pths)
    plt.savefig("img/sim_env3.png", dpi=300)

if __name__ == "__main__":
    c240828()


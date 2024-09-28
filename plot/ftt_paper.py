import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif",
    # "font.family": "Times New Roman",
    "font.sans-serif": "Helvetica",
})

color_default = ["#0050E6", "#f57600", "#AD3131", "#B68961", "#009eb0", "#A9B388", "#D8ABAB"]
dataset_name = {
    "medexp": "Medium-Expert",
    "medrep": "Medium-Replay",
    "medium": "Medium",
    "random": "Random"
}

def formated_learning_curve(axs, pths, envs, datasets, colors={}, styles={}, show_title=True, highlight=False):
    handle_list, label_list = [], []
    for i, e in enumerate(envs):
        for j, d in enumerate(datasets):
            ed_pth = fill_in_path(pths, [e, d])
            learning_curve(axs[i][j], ed_pth, 10, colors, styles, highlight=highlight)
            if show_title:
                axs[i][j].set_title("{}\n{}".format(e, dataset_name[d]))
            axs[i][j].spines[['right', 'top']].set_visible(False)

            handles, labels = axs[i][j].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_cp = copy.copy(handle)
                    handle_cp.set_alpha(1)
                    handle_list.append(handle_cp)
                    label_list.append(label)
    return axs, handle_list, label_list

def projection_baseline():
    envs = ["HalfCheetah"]
    # datasets = ["medexp", "medium"]
    datasets = ["medexp"]
    pths = {
        "FtT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "ReverseKL": "../data/output/test_v0/naive_projection_baselines/{}/{}/TAWACqG/qGaussian/",
        "RAR": "../data/output/test_v0/naive_projection_baselines/{}/{}/TAWACqGC/qGaussian/",
        # "SPOT": "../data/output/test_v1/baseline/{}/{}/SPOT/SGaussian/",
    }
    colors = {
        "FtT": color_default[0],
        "ReverseKL": color_default[1],
        "RAR": color_default[2]
    }
    fig, axs = plt.subplots(len(envs), len(datasets),
                            figsize=(2.5 * len(datasets), 2 * len(envs)))
    if len(envs) == 1:
        axs = np.array([axs])
    axs = axs.reshape(len(envs), len(datasets))
    axs, handle_list, label_list = formated_learning_curve(axs, pths, envs, datasets, colors)
    hc_y = [0, 0.4, 0.8]
    axs[0, 0].set_yticks(hc_y)
    axs[0, 0].set_yticklabels([int(y * 100) for y in hc_y])
    # hc_y = [0, 0.2, 0.4]
    # axs[0, 1].set_yticks(hc_y)
    # axs[0, 1].set_yticklabels([int(y * 100) for y in hc_y])

    fontsize = 11
    fig.supxlabel(r'Step $(\times 10^4)$', y=0, verticalalignment='bottom', fontsize=fontsize) # discrete 10^3, continuous 10^4
    fig.supylabel('Score', x=0.02, y=0.4, verticalalignment='bottom', fontsize=fontsize)
    fig.text(0.77, 0.61, "FtT", fontsize=fontsize, color=colors["FtT"])
    fig.text(0.6, 0.25, "ReverseKL", fontsize=fontsize, color=colors["ReverseKL"])
    fig.text(0.77, 0.46, "RAR", fontsize=fontsize, color=colors["RAR"])

    fig.tight_layout(rect=[-0.1, -0.15, 1, 1])
    # fig.supxlabel(r'Step $(\times 10^4)$', y=0, verticalalignment='bottom', fontsize=12) # discrete 10^3, continuous 10^4
    # fig.supylabel('Score', x=0.05, y=0.4, verticalalignment='bottom', fontsize=12)
    # fig.legend(handle_list, label_list, ncol=3, loc='upper center',
    #            bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=13)
    # fig.tight_layout(rect=[0, -0.15, 1, 0.93])
    plt.savefig("img/projection_baseline.pdf", dpi=300)

def main_results():
    # envs = ["HalfCheetah"]
    # datasets = ["medexp"]
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medrep", "medium"]
    pths = {
        "FTT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        # "FTT-SG": "../data/output/test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        # "FTT-SPOT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/{}/{}/FTT/qGaussian/",
        # "IQL-HTqG": "../data/baseline_data/output/test_v1/{}/{}/IQL/HTqGaussian/",
        "IQL": "../data/baseline_data/output/test_v1/{}/{}/IQL/SGaussian/",
        # "InAC-HTqG": "../data/baseline_data/output/test_v1/{}/{}/InAC/HTqGaussian/",
        "InAC": "../data/baseline_data/output/test_v1/{}/{}/InAC/SGaussian/",
        # "TAWAC-HTqG": "../data/baseline_data/output/test_v1/{}/{}/TAWAC/HTqGaussian/",
        "TAWAC": "../data/baseline_data/output/test_v1/{}/{}/TAWAC/SGaussian/",
        "AWAC": "../data/baseline_data/output/test_v1/{}/{}/AWAC/SGaussian/",
        "XQL": "../data/output/test_v1/baseline/{}/{}/XQL/SGaussian/",
        "SQL": "../data/output/test_v1/baseline/{}/{}/SQL/SGaussian/",
        # "SPOT": "../data/output/test_v1/baseline/{}/{}/SPOT/SGaussian/",
    }
    colors = {
        "FTT": color_default[0],
        "FTT-SG": 'black',
        "FTT-SPOT": 'black',
        "IQL": color_default[1],
        "InAC": color_default[2],
        "TAWAC": color_default[3],
        "AWAC": color_default[4],
        "XQL": color_default[5],
        "SQL": color_default[6],
        "SPOT": 'grey'
    }
    fig, axs = plt.subplots(len(envs), len(datasets),
                            figsize=(2.8 * len(datasets), 2 * len(envs)))
    # axs = np.asarray([[axs]])
    axs, handle_list, label_list = formated_learning_curve(axs, pths, envs, datasets, colors, highlight="FTT")
    hc_y = [0, 0.5, 1]
    for row in range(len(envs)):
        axs[row, 0].set_yticks(hc_y)
        axs[row, 0].set_yticklabels([int(y * 100) for y in hc_y])
    hc_y = [0, 0.2, 0.4]
    for col in range(1, 3):
        axs[0, col].set_yticks(hc_y)
        axs[0, col].set_yticklabels([int(y * 100) for y in hc_y])
    hc_y = [0, 0.35, 0.7]
    for row in range(1, 3):
        for col in range(1, 3):
            axs[row, col].set_yticks(hc_y)
            axs[row, col].set_yticklabels([int(y * 100) for y in hc_y])

    fig.supxlabel(r'Step $(\times 10^4)$', y=0.05, verticalalignment='bottom', fontsize=13) # discrete 10^3, continuous 10^4
    fig.supylabel('Score', x=0.02, y=0.4, verticalalignment='bottom', fontsize=13)
    fig.legend(handle_list, label_list, ncol=7, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=11)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("img/main_results.pdf", dpi=300)

def simulation_exp():
    envs = ["SimEnv3"]
    datasets = ["random"]
    pths = {
        "FTT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        # "FTT-SG": "../data/output/test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "IQL": "../data/output/test_v1/baseline/{}/{}/IQL/SGaussian/",
        "XQL": "../data/output/test_v1/baseline/{}/{}/XQL/SGaussian/",
        "SQL": "../data/output/test_v1/baseline/{}/{}/SQL/SGaussian/",
        # "SPOT": "../data/output/test_v1/baseline/{}/{}/SPOT/SGaussian/",
        # "TAWAC": "../data/output/test_v1/baseline/{}/{}/TAWAC/SGaussian/",
    }
    colors = {
        "FTT": color_default[0],
        "IQL": color_default[1],
        "XQL": color_default[2],
        "SQL": color_default[3],
        "SPOT": color_default[4],
        # "FTT-SG": color_default[1],
        # "TAWAC": color_default[3],
    }
    fig, axs = plt.subplots(len(envs), len(datasets), figsize=(3, 3))
    axs = np.array([[axs]])
    axs, handle_list, label_list = formated_learning_curve(axs, pths, envs, datasets, colors, show_title=False)
    fig.text(0.4, 0.85, "Learning Curve", fontsize=12)
    fig.supxlabel(r'Step $(\times 10^4)$', y=0.05, verticalalignment='bottom', fontsize=13) # discrete 10^3, continuous 10^4
    fig.supylabel('Score', x=0.02, y=0.4, verticalalignment='bottom', fontsize=13)
    fig.legend(handle_list, label_list, ncol=2, loc='lower left',
               bbox_to_anchor=(0.25, 0.21), frameon=False, fontsize=10)
    fig.tight_layout()
    plt.subplots_adjust(top=0.8, left=0.2, bottom=0.2)
    # plt.subplots_adjust(top=0.75, left=0.2, bottom=0.15)
    # plt.subplots_adjust(bottom=0.23, left=0.2)
    plt.savefig("img/simenv.pdf", dpi=300)

def spot():
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medrep", "medium"]
    pths = {
        "FTT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "FTT-SPOT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/{}/{}/FTT/qGaussian/",
        "FTT-SG": "../data/output/test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
    }
    colors = {
        "FTT": color_default[0],
        "FTT-SPOT": color_default[1],
        "FTT-SG": color_default[2],
    }
    fig, axs = plt.subplots(len(envs), len(datasets),
                            figsize=(2.8 * len(datasets), 2 * len(envs)))
    axs, handle_list, label_list = formated_learning_curve(axs, pths, envs, datasets, colors)
    hc_y = [0, 0.5, 1]
    for row in range(len(envs)):
        axs[row, 0].set_yticks(hc_y)
        axs[row, 0].set_yticklabels([int(y * 100) for y in hc_y])
    hc_y = [0, 0.2, 0.4]
    for col in range(1, 3):
        axs[0, col].set_yticks(hc_y)
        axs[0, col].set_yticklabels([int(y * 100) for y in hc_y])
    hc_y = [0, 0.35, 0.7]
    for row in range(1, 3):
        for col in range(1, 3):
            axs[row, col].set_yticks(hc_y)
            axs[row, col].set_yticklabels([int(y * 100) for y in hc_y])

    fig.supxlabel(r'Step $(\times 10^4)$', y=0.05, verticalalignment='bottom', fontsize=13) # discrete 10^3, continuous 10^4
    fig.supylabel('Score', x=0.02, y=0.4, verticalalignment='bottom', fontsize=13)
    fig.legend(handle_list, label_list, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=13)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("img/kl_spot.pdf", dpi=300)

def tawac_htqg():
    pths = {
        "FTT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        # "FTT-SPOT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/{}/{}/FTT/qGaussian/",
        "TAWAC-HT": "../data/baseline_data/output/test_v1/{}/{}/TAWAC/HTqGaussian/",
        # "InAC-HT": "../data/baseline_data/output/test_v1/{}/{}/InAC/HTqGaussian/",
    }
    colors = {
        "FTT": color_default[0],
        # "FTT-SPOT": color_default[2],
        "TAWAC-HT": color_default[1],
        "InAC-HT": color_default[3],
    }
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medrep", "medium"]
    fig, axs = plt.subplots(len(envs), len(datasets),
                            figsize=(2.8 * len(datasets), 2 * len(envs)))
    axs, handle_list, label_list = formated_learning_curve(axs, pths, envs, datasets, colors)
    hc_y = [0, 0.5, 1]
    for row in range(len(envs)):
        axs[row, 0].set_yticks(hc_y)
        axs[row, 0].set_yticklabels([int(y * 100) for y in hc_y])
    hc_y = [0, 0.2, 0.4]
    for col in range(1, 3):
        axs[0, col].set_yticks(hc_y)
        axs[0, col].set_yticklabels([int(y * 100) for y in hc_y])
    hc_y = [0, 0.35, 0.7]
    for row in range(1, 3):
        for col in range(1, 3):
            axs[row, col].set_yticks(hc_y)
            axs[row, col].set_yticklabels([int(y * 100) for y in hc_y])

    fig.supxlabel(r'Step $(\times 10^4)$', y=0.05, verticalalignment='bottom', fontsize=13) # discrete 10^3, continuous 10^4
    fig.supylabel('Score', x=0.02, y=0.4, verticalalignment='bottom', fontsize=13)
    fig.legend(handle_list, label_list, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=13)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("img/ftt_httawac.pdf", dpi=300)

def final_perf_bar(ax, pths, envs, datasets, colors, baseline):
    width = 1.0 / (len(datasets) + 1)
    final_perf = {a:[] for a in pths}
    x_axis = []
    for e in envs:
        for d in datasets:
            ed_pth = fill_in_path(pths, [e, d])
            perf = load_final_perf(ed_pth, 10)
            for a in perf:
                final_perf[a].append(perf[a])
            # x_axis.append("{} {}".format(e, formal_dataset_name[d]))
            x_axis.append("{}".format(formal_dataset_name[d]))
    base_perf = np.asarray(final_perf[baseline])
    proportions = {}
    x_coord = np.arange(len(x_axis))
    for a in final_perf:
        if a == baseline:
            continue
        proportions[a] = np.asarray(final_perf[a]) / base_perf

    for i, a in enumerate(proportions):
        ax.bar(x_coord+i*width, proportions[a], width=width, color=colors[a], label=a)
    num_area_fill = len(envs) // 2
    for area in range(num_area_fill):
        start_x = x_coord[(area * 2 + 1) * len(datasets)] - 0.5 + width*0.5
        end_x = x_coord[(area * 2 + 2) * len(datasets)] - 0.5 + width*0.5
        ax.axvspan(start_x, end_x, alpha=.2, facecolor='grey', edgecolor=None)

    ax.axhline(1, color='grey', linestyle='--', linewidth=0.5)
    ax.set_xticks(x_coord+len(proportions)//2 * width - 0.5*width)
    ax.set_xticklabels(x_axis, rotation=40, ha='right')
    for i, e in enumerate(envs):
        x = x_coord[len(x_coord) // len(envs)*i + len(datasets)//2]
        ax.text(x, 1.15, e, ha='center', va='center', fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)
    return ax

def ablation():
    pths = {
        "FTT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "FTT-SG": "../data/output/test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/{}/{}/FTT/qGaussian/",
        "FTT-SPOT": "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/{}/{}/FTT/qGaussian/",
        "TAWAC-HT": "../data/baseline_data/output/test_v1/{}/{}/TAWAC/HTqGaussian/",
    }
    colors = {
        "FTT-SPOT": color_default[0],
        "TAWAC-HT": color_default[1],
        "FTT-SG": color_default[2],
    }
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medrep", "medium"]
    fig, ax = plt.subplots(1, 1, figsize=(0.5*len(datasets)*len(envs)+2, 2.5))
    ax = final_perf_bar(ax, pths, envs, datasets, colors, baseline="FTT")
    ax.set_ylabel('Final Performance\nPropotional to FtT '+r'$(\%)$')

    ys = [0.6, 0.8, 1.0, 1.2]
    ax.set_yticks(ys)
    ax.set_yticklabels([y*100 for y in ys])
    ax.set_ylim(0.6, 1.2)

    # plt.legend(loc='lower left', bbox_to_anchor=(0.1, 1.), frameon=False, ncol=2)
    plt.legend(loc='lower left', bbox_to_anchor=(1., 0.2), frameon=False, ncol=1)
    fig.tight_layout()
    plt.savefig("img/ablation.pdf", dpi=300)


if __name__ == "__main__":
    # projection_baseline()
    # main_results()
    # simulation_exp()
    # ablation()
    spot()
    # tawac_htqg()

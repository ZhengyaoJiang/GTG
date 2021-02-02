import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Analyse the logs produced by torchbeast")

parser.add_argument("--dir", type=str, default="~/logs/torchbeast", help="Directory for log files.")
parser.add_argument("--mode", type=str, default="table", choices=["table", "plot", "joint_plot", "group_plot"])
parser.add_argument("--idx", "--index", nargs="+", required=True)
parser.add_argument("--repeats", default=3, type=int)
parser.add_argument("--steps", default=float('inf'), type=float)
parser.add_argument("--baseline", default=0.0, type=float)
parser.add_argument("--labels",nargs="+")
parser.add_argument("--name", type=str, default="")


def joint_plot(indexes, labels, dir, steps, name):
    plt.figure()
    dfs = []
    for index, label in zip(indexes, labels):
        df = pd.read_csv(os.path.join(dir, index, "logs.csv"), index_col="# _tick")
        df = df[df["step"] < steps]
        df["label"] = label
        df["smoothed return"] = df["mean_episode_return"].ewm(span=200).mean()
        dfs.append(df)
    data = pd.concat(dfs)
    sns.relplot(x="step", y="smoothed return", kind="line",
                data=data, aspect=1.5, hue="label")
    if not name:
        name = os.path.expanduser(os.path.join(dir, "".join(indexes) + "return.png"))
    else:
        name = os.path.expanduser(os.path.join(dir, name))
    Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
    plt.savefig(name)

def group_plot(indexes, labels, dir, steps, name, repeats=3, summary="mean", baseline=0.0):
    fig = plt.figure(figsize=(2.3, 2.3))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.gcf().subplots_adjust(bottom=0.17)
    summary_lines = []
    for group_n,(index, label) in enumerate(zip(indexes, labels)):
        in_group = []
        for i in range(repeats):
            df = pd.read_csv(os.path.join(dir, f"{index}-{i+1}", "logs.csv"), index_col="# _tick")
            df = df[df["step"]<steps]
            df["label"] = label
            df["id"] = f"{index}-{i+1}"
            df["size"] = 0.1
            df["smoothed return"] = df["mean_episode_return"].ewm(span=2000).mean()
            step = df["step"]
            returns = df["smoothed return"]
            plt.plot(step, returns, f"C{group_n}", alpha=0.3, linewidth=0.3)
            in_group.append([step, returns, np.array(returns)[-1]])
        if summary == "median":
            final_returns = [in_group[i][-1] for i in range(repeats)]
            median = in_group[np.argsort(final_returns)[len(final_returns) // 2]]
            summary_lines.append(plt.plot(median[0], median[1], f"C{group_n}", linewidth=1.5)[0])
        elif summary == "mean":
            mean = [np.mean(pd.DataFrame([x[0] for x in in_group]), axis=0),
                    np.mean(pd.DataFrame([x[1] for x in in_group]), axis=0)]
            summary_lines.append(plt.plot(mean[0], mean[1], f"C{group_n}", linewidth=1.5)[0])
    if baseline != 0.0:
        plt.axhline(baseline, color="r", linestyle="dashed")
    plt.xlabel('step')
    plt.ylabel('smoothed return')
    plt.legend(summary_lines, labels, prop={'size': 7})
    if not name:
        name = os.path.expanduser(os.path.join(dir, "".join(indexes)+"return.png"))
    else:
        name = os.path.expanduser(os.path.join(dir, name))
    Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
    plt.savefig(name)


def plot_mean_return(index, dir):
    plt.figure()
    df = pd.read_csv(os.path.join(dir, index, "logs.csv"), index_col="# _tick")
    df["smoothed return"] = df["mean_episode_return"].ewm(span=100).mean()
    sns.relplot(x="step", y="smoothed return", kind="line", data=df, aspect=2.0)
    plt.savefig(os.path.expanduser(os.path.join(dir, index, "return.png")))

def compute_return_stats(indexes, dir):
    result = {}
    for i in indexes:
        df = pd.read_csv(os.path.join(dir,i,"logs.csv"), index_col="# _tick")
        last = df["mean_episode_return"].iloc[-500:]
        result[i]=(last.mean(), last.std())
    return result

def print_stats_dict(stats):
    for i, stat in stats.items():
        print(f"{i}: {stat[0]:.2f} ± {stat[1]:.2f}")

def main(flags):
    if flags.mode == "table":
        print_stats_dict(compute_return_stats(flags.idx, flags.dir))
    if flags.mode == "plot":
        for idx in flags.idx:
            plot_mean_return(idx, flags.dir)
    if flags.mode == "joint_plot":
        joint_plot(flags.idx, flags.labels, flags.dir, flags.steps, flags.name)
    if flags.mode == "group_plot":
        group_plot(flags.idx, flags.labels, flags.dir, flags.steps, flags.name,
                   repeats=flags.repeats, baseline=flags.baseline)



if __name__ == "__main__":
    flags = parser.parse_args(
        
    )
    main(flags)

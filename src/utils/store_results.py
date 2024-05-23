import os
from collections import defaultdict
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap

from utils.config import Config, save_config

def store_results(df, config: Config):
    if config.plot.use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman",
            "text.latex.preamble":  r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}" "\n" r"\usepackage{amsfonts}",
        })

    hatches = defaultdict(lambda: None)
    hatches.update(**{"IPS": "**", "DM": "..", "DR": "OO"})

    xvar = config.plot.xvar
    yvar = config.plot.yvar
    
    directory = config.experiment.directory
    filename = datetime.isoformat(datetime.now()).replace(":", "-")
    os.makedirs(f"results/data/{directory}", exist_ok=True)
    os.makedirs(f"results/figures/{directory}", exist_ok=True)
    os.makedirs(f"results/configs/{directory}", exist_ok=True)
    df.to_csv(f"results/data/{directory}/{filename}.csv", index=False)
    save_config(config, f"results/configs/{directory}/{filename}.yaml")

    if xvar is not None:
        unique_xvars = list(df[xvar].dropna().unique())
        unique_xvars_np = np.array(unique_xvars)
        if unique_xvars_np.dtype.kind == "f" and np.all(unique_xvars_np == unique_xvars_np.astype(int)):
            unique_xvars = unique_xvars_np.astype(int).tolist()
        df[xvar] = df[xvar].map(lambda k: unique_xvars.copy() if pd.isna(k) else [k])
        results = df.explode(xvar)
    else:
        results = df

    def gmean(arr):
        """Geometric mean"""
        if type(arr) == pd.DataFrame:
            arr = arr[yvar].values
        if type(arr) == pd.Series:
            arr = arr.values
        return np.exp(np.log(arr).mean())    

    def se(arr):
        """Empirical standard error"""
        if type(arr) == pd.DataFrame or type(arr) == pd.Series:
            arr = arr[yvar].values
        return bootstrap((arr,), gmean, confidence_level=0.68)

    
    if config.plot.type == "lineplot":
        if xvar is None:
            xvar = next(iter(config.experiment.ablation.keys()))
        unique_xvars = list(df[xvar].dropna().unique())
        fig, ax = plt.subplots(figsize=config.plot.figsize)
        sns.lineplot(data=results, x=xvar, y=yvar, ax=ax, hue="Estimator", markers=True, style="Estimator", estimator=gmean)
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        
        plt.yscale(config.plot.yscale)
        plt.xscale(config.plot.xscale)
        plt.xlabel(config.plot.xlabel)
        plt.ylabel(config.plot.ylabel)
        plt.title(config.plot.title)
        ax.set_xticks(sorted(unique_xvars))
        ax.set_xticklabels(sorted(unique_xvars))
        plt.savefig(f"results/figures/{directory}/{filename}.pdf", bbox_inches="tight")
        
    elif config.plot.type == "table":
        groups = ["Estimator", xvar] if xvar is not None else ["Estimator"]
        mean_results = results.groupby(groups).apply(gmean)
        se_results = results.groupby(groups).apply(se)
        statistic_results = pd.concat([mean_results, se_results], axis=1)
        print("Mean and standard error")
        print(statistic_results.apply(lambda row: f"{row[0]:.1e} ± {row[0] - row[1].confidence_interval.low:.1e}", axis=1)) 
        
    elif config.plot.type == "catplot":
        if xvar is None:
            xvar = next(iter(config.experiment.ablation.keys()))
        unique_xvars = list(df[xvar].dropna().unique())
        results[xvar] = results[xvar].str.replace("\"", "")
        g = sns.catplot(data=results, x=xvar, y=yvar, hue="Estimator", kind="bar", aspect=3, height=3, estimator=gmean)
        if g._legend is not None:
            g._legend.remove()
        
        unique_hues = results["Estimator"].unique()
        unique_bars = list(product(unique_hues, unique_xvars))
        edge_colors = {}
        face_colors = {}
        
        # Add hatches to bars
        for patch, bar in zip(g.ax.patches, unique_bars):
            hue, x = bar
            patch.set_hatch(hatches[hue])
            patch.set_edgecolor(patch.get_facecolor())
            if hatches[hue] is not None:
                patch.set_facecolor("none")
            edge_colors[hue] = patch.get_edgecolor()
            face_colors[hue] = patch.get_facecolor()
                          
        # Create legend handles manually
        legend_patches = [matplotlib.patches.Patch(edgecolor=edge_colors[label], facecolor=face_colors[label], hatch=hatches[label], label=label) for label in unique_hues]
        plt.legend(handles=legend_patches, title="", bbox_to_anchor=(0.5, 1.07), loc="upper center", ncol=len(unique_hues), frameon=False)        

        plt.yscale(config.plot.yscale)
        plt.xlabel(config.plot.xlabel)
        plt.ylabel(config.plot.ylabel)
        plt.title(config.plot.title)
        plt.savefig(f"results/figures/{directory}/{filename}.pdf", bbox_inches="tight")
        
    elif config.plot.type == "tuningplot":
        results[["Estimator", "Tuning"]] = results["Estimator"].str.split(pat=" ", n=1).apply(pd.Series)
        sns.set_palette(sns.color_palette()[2:])
        g = sns.catplot(data=results, x="Estimator", y=yvar, hue="Tuning", kind="bar", aspect=3, height=3, estimator=gmean)
        if g._legend is not None:
            g._legend.remove()
            
        plt.legend(title="", bbox_to_anchor=(0.5, 1.07), loc="upper center", ncol=len(results["Tuning"].unique()), frameon=False)
        
        plt.yscale(config.plot.yscale)
        plt.xlabel(config.plot.xlabel)
        plt.ylabel(config.plot.ylabel)
        plt.title(config.plot.title)
        plt.savefig(f"results/figures/{directory}/{filename}.pdf", bbox_inches="tight")

    elif config.plot.type == "barplot":
        if xvar is None:
            xvar = "Estimator"
        fig, ax = plt.subplots(figsize=config.plot.figsize)
        g = sns.barplot(data=results, x=xvar, y=yvar, estimator=gmean, ax=ax, dodge=False)

        # Remove x-axis labels
        plt.gca().axes.get_xaxis().set_visible(False)

        # Add hatches to bars
        unique_hues = results[xvar].unique()
        for i, patch in enumerate(g.patches):
            hue = unique_hues[i]
            patch.set_hatch(hatches[hue])
            patch.set_edgecolor(patch.get_facecolor())
            if hatches[hue] is not None:
                patch.set_facecolor("none")

        # Get the unique values from the hue column and the colors used in the plot
        edge_colors = [patch.get_edgecolor() for patch in g.patches]
        face_colors = [patch.get_facecolor() for patch in g.patches]

        # Create legend handles manually
        legend_patches = [matplotlib.patches.Patch(edgecolor=edge, facecolor=face, hatch=hatches[label], label=label) for edge, face, label in zip(edge_colors, face_colors, unique_hues)]
        plt.legend(handles=legend_patches, bbox_to_anchor=(1, 0.5), loc="center left", frameon=False)

        plt.yscale(config.plot.yscale)
        plt.ylabel(config.plot.ylabel)
        plt.title(config.plot.title)
        plt.savefig(f"results/figures/{directory}/{filename}.pdf", bbox_inches="tight")
    
    print("Average time for each method")
    print(results.groupby("Estimator").mean(numeric_only=True)["seconds"].sort_values(ascending=False).apply("{:.2f}s".format))

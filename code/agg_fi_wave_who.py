# Purpose: Aggregate Feature Importance of XGBoost Model, Plot
# Inputs: feature importances file from XGBoost.py file, for all outcomes and features
# Outputs: Plots of feature importance aggregated by wave and respondent
# Machine: Laptop, <1 min


import numpy as np
import re
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(0)


pat = re.compile("^(kind|hv|f|k|m|o|p|t)(\d)")

if __name__ == "__main__":
    ### INPUT ###
    df = pd.read_csv("../output/7_feature_importances.csv", index_col=0)
    ### INPUT ###

    df.loc[:, "who"] = df.index.tolist()
    who_list = []
    wave_list = []
    for who in df["who"]:
        if who[0] == "c":
            who = who[1:]
        if who.find("kind") == 0:
            symbol = "kind"
        elif who.find("hv") == 0:
            symbol = "hv"
        elif who[0] in ["f", "k", "m", "o", "p", "t"]:
            symbol = who[0]
        else:
            symbol = None
        who_list.append(symbol)
        m = pat.match(who)
        if m:
            wave_list.append(int(m.group(2)))
        else:
            wave_list.append("N/A")

    df.loc[:, "who"] = who_list
    df.loc[:, "wave"] = wave_list

    columns = ["gpa", "eviction", "grit", "materialHardship", "jobTraining", "layoff"]

    ### OUTPUT ###
    df.groupby("wave").sum()[columns].to_csv("../output/7_fi_wave.csv")
    df.groupby("wave").sum().plot(kind="bar")
    plt.savefig("../output/fig/7_fi_wave.png")
    plt.close()

    df.groupby("who").sum()[columns].to_csv("../output/7_fi_who.csv")
    df.groupby("who").sum().plot(kind="bar")
    plt.savefig("../output/fig/7_fi_who.png")
    plt.close()
    ### OUTPUT ###

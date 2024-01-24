import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def smooth(csv_path, weight=0.85):
    data = pd.read_csv(
        filepath_or_buffer=csv_path,
        header=0,
        names=["Step", "Value"],
        dtype={"Step": np.int, "Value": np.float},
    )
    scalar = data["Value"].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({"Step": data["Step"].values, "Value": smoothed})
    save.to_csv("smooth_" + csv_path)


def smooth_and_plot(csv_path, weight=0.85):
    data = pd.read_csv(
        filepath_or_buffer=csv_path,
        header=0,
        names=["Step", "Value"],
        dtype={"Step": np.int, "Value": np.float},
    )
    scalar = data["Value"].values
    last = scalar[0]
    print(type(scalar))
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    # save = pd.DataFrame({'Step':data['Step'].values, 'Value':smoothed})
    # save.to_csv('smooth_' + csv_path)

    steps = data["Step"].values
    steps = steps.tolist()
    origin = scalar.tolist()

    fig = plt.figure(1)
    plt.plot(steps, origin, label="origin")
    plt.plot(steps, smoothed, label="smoothed")
    # plt.ylim(0, 220) # Tensorboard中会滤除过大的数据，可通过设置坐标最值来实现
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # smooth('total_loss.csv')
    smooth_and_plot("total_loss.csv")

import math

import matplotlib.pyplot as plt
import csv
import numpy as np

def plot():
    iterations = []
    avg_points = []
    avg_rewards = []

    with open('stats.csv') as file:
        data = csv.reader(file, delimiter=',')
        for row in data:
            iterations.append(row[0])
            avg_points.append(row[1])
            avg_rewards.append(row[7])

    iterations = np.array(iterations).astype(np.int64)
    avg_points = np.array(avg_points).astype(np.double)
    avg_rewards = np.array(avg_rewards).astype(np.double).clip(-1000, math.inf)

    plt.plot(iterations, avg_points, color="forestgreen")
    plt.xlabel("iteration")
    plt.ylabel("avg points over 100 rounds")
    plt.title("Average Points During Training")
    plt.grid()
    plt.show()

    plt.plot(iterations, avg_rewards, color="orangered")
    plt.xlabel("iteration")
    plt.ylabel("avg rewards over 100 rounds")
    plt.title("Average rewards During Training")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot()

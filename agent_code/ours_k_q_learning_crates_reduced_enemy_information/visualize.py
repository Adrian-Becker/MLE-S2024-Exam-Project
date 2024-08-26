import matplotlib.pyplot as plt
import csv
import numpy as np

def plot():
    iterations = []
    avg_points = []
    avg_bombs = []

    with open('stats.csv') as file:
        data = csv.reader(file, delimiter=',')
        for row in data:
            iterations.append(row[0])
            avg_points.append(row[1])
            avg_bombs.append(row[2])

    iterations = np.array(iterations).astype(np.int64)
    avg_points = np.array(avg_points).astype(np.double)
    avg_bombs = np.array(avg_bombs).astype(np.double)

    plt.plot(iterations, avg_points, color="forestgreen")
    plt.xlabel("iteration")
    plt.ylabel("avg points over 100 rounds")
    plt.title("Average Points During Training")
    plt.show()

    plt.plot(iterations, avg_bombs, color="orangered")
    plt.xlabel("iteration")
    plt.ylabel("avg bombs over 100 rounds")
    plt.title("Average Bombs During Training")
    plt.show()


if __name__ == "__main__":
    plot()

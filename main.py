import random
import numpy as np
from kohonen import Kohonen

def Part_A_1():
    points = []
    for i in range(1000):
        points.append([random.uniform(0, 1), random.uniform(0, 1)])
    layers = (np.ones(10) * 10).astype(int)
    ko = Kohonen()
    ko.fit(points)
    ko2 = Kohonen(neurons_amount=layers)
    ko2.fit(points)
    return

def Part_A_2():
    points = []
    for i in range(1000):
        points.append([random.gauss(0.5, 0.15), random.uniform(0, 1)])
    layers = (np.ones(10) * 10).astype(int)
    ko = Kohonen(neurons_amount=layers)
    ko.fit(points)

    points = []
    for i in range(1000):
        points.append([random.gauss(0.5, 0.15), random.gauss(0.5, 0.15)])
    ko = Kohonen(neurons_amount=layers)
    ko.fit(points)
    return

def Part_A_3():
    points = []
    for i in range(7000):
        points.append([random.uniform(0, 1), random.uniform(0, 1)])
    circle = []
    for p in points:
        if 0.15**2 <= (p[0] - 0.5)**2 + (p[1]-0.5)**2 <= 0.3**2:
            circle.append(p)
    layers = (np.ones(10) * 10).astype(int)
    ko = Kohonen(neurons_amount=[30], learning_rate=0.05)
    ko.fit(circle, iteration=10000)
    ko2 = Kohonen(neurons_amount=layers, learning_rate=0.5)
    ko2.fit(circle)
    pass

def Part_B():

    pass

if __name__ == '__main__':
    Part_A_1()
    Part_A_2()
    Part_A_3()
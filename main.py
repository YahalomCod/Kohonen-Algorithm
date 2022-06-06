import random
import numpy as np
from kohonen import Kohonen

# def Part_A():
#     points = []
#     for i in range(1000):
#         points.append([random.uniform(0, 1), random.uniform(0, 1)])
#
#     pass

if __name__ == '__main__':
    # points = []
    # for i in range(1000):
    #     points.append([random.uniform(0, 1), random.uniform(0, 1)])
    # print(points)
    # ko = Kohonen()
    # ko.fit(points)
    # print(ko.lamda)
    # print(ko.radius)
    # print(ko.learning_rate)
    # print(ko.neurons)
    # ko.plot1D()

    # points = []
    # for i in range(1000):
    #     points.append([random.uniform(0, 1), random.uniform(0, 1)])
    # print(points)
    # layers = (np.ones(10)*10).astype(int)
    # print(layers)
    # ko = Kohonen(neurons_amount=layers)
    # ko.fit(points)
    # print(ko.lamda)
    # print(ko.radius)
    # print(ko.learning_rate)
    # print(ko.neurons[:,:,0])

    # ko.plot2D()

    points = []
    for i in range(10000):
        points.append([i/1000 , random.uniform(0, 1)])
    circle = []
    for p in points:
        if 0.25<(p[0]-0.5)**2 + (p[0]-0.5)**2 < 0.5 :
            circle.append(p)

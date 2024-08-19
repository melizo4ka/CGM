import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math
from scipy.linalg import solve
import pandas as pd
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)


def plotting(k, der_gd, der_cg, download, folder='plots'):
    plt.plot(k, der_gd, marker='.', color='Cyan', label="Gradient Descent")
    plt.plot(k, der_cg, marker='.', color='Magenta', label="Conjugate Gradient")
    plt.xlabel('Iterations')
    plt.ylabel('Norm of the derivative')
    plt.legend(loc="upper right")

    if download:
        if not os.path.exists(folder):
            os.makedirs(folder)

        timestamp = datetime.now().strftime('%H%M%S') + f"{datetime.now().microsecond // 1000:03d}"
        filename = f'plot_{timestamp}.png'
        filepath = os.path.join(folder, filename)
        plt.savefig(filepath)

    else:
        plt.show()

    plt.clf()


def generate(dim):
    a_matrix = np.random.uniform(-1, 1, (dim, dim))
    c_vector = np.random.uniform(-1, 1, dim)
    return a_matrix, c_vector


def f(x, Q, c):
    return 0.5 * np.dot(np.transpose(x), np.dot(Q, x)) + np.dot(np.transpose(c), x)


def gradient(Q, x, c):
    grad = np.dot(Q, x) + c
    return grad


def direction(Q, x, c):
    dk = - gradient(Q, x, c)
    return dk


def step_GD(Q, x, c):
    dk = direction(Q, x, c)
    ak = -(np.dot(gradient(Q, x, c), dk)) / (np.dot(np.transpose(dk), np.dot(Q, dk)))
    return ak


def gradient_descent(Q, c, dim, ths, iter_on_dim):
    if iter_on_dim:
        condition = lambda: k < dim
    else:
        condition = lambda: k < dim ** 3

    x = np.zeros(dim)
    history_x = [np.copy(x)]
    k_values = []
    der_values = []
    ths_values = []
    k = 0

    while condition():
        dk = direction(Q, x, c)
        ak = step_GD(Q, x, c)
        x = x + ak * dk
        der_values.append(np.linalg.norm(gradient(Q, x, c)))

        for _, th in enumerate(ths):
            if not any(t[0] == th for t in ths_values):
                if np.linalg.norm(gradient(Q, x, c)) < th:
                    ths_values.append((th, k))

        history_x.append(x.copy())
        k += 1
        k_values.append(k)

    if not iter_on_dim:
        df = pd.DataFrame(ths_values, columns=['Threshold', 'Num of iterations'])
        print(df.to_string(index=False))

    return history_x, k_values, der_values


def step_CG(gk, Q, dk):
    ak = (pow(np.linalg.norm(gk), 2)) / (np.dot(np.transpose(dk), np.dot(Q, dk)))
    return ak


def conjugate_gradient(Q, c, dim, ths, iter_on_dim):
    if iter_on_dim:
        condition = lambda: k < dim
    else:
        condition = lambda: k < dim ** 3

    x = np.zeros(dim)
    history_x = [np.copy(x)]
    der_values = []
    gk = np.dot(Q, x) + c
    dk = - gk
    ths_values = []
    k = 0

    while condition():
        ak = step_CG(gk, Q, dk)
        x = x + ak * dk
        der_values.append(np.linalg.norm(gradient(Q, x, c)))

        for _, th in enumerate(ths):
            if not any(t[0] == th for t in ths_values):
                if np.linalg.norm(gradient(Q, x, c)) < th:
                    ths_values.append((th, k))

        history_x.append(x.copy())

        gk_new = gk + np.dot(ak, np.dot(Q, dk))
        bk = (np.linalg.norm(gk_new) ** 2) / (np.linalg.norm(gk) ** 2)
        dk = - gk_new + bk * dk
        gk = gk_new
        k += 1

    if not iter_on_dim:
        df = pd.DataFrame(ths_values, columns=['Threshold', 'Num of iterations'])
        print(df.to_string(index=False))

    return history_x, der_values


def check_solution(Q, c, x_algo):
    x = solve(Q, -c)
    if np.all(x == x_algo):
        print("The result of the algorithm is correct")
    else:
        print("The mean of the difference is", np.mean(x - x_algo))


def condition_number(matrix):
    return np.linalg.cond(matrix)


if __name__ == '__main__':
    # possible dimensions of the problem
    dimensions = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75]

    thresholds = [math.pow(10, -1), math.pow(10, -2), math.pow(10, -3),
                  math.pow(10, -5), math.pow(10, -7)]

    # True if you want to iterate for the dimension of the problem, False if dim^3 (used to get better thresholds)
    iterate_on_dim = True

    for _, dimension in enumerate(dimensions):
        A, c = generate(dimension)
        Q = np.dot(A, np.transpose(A))
        # print("The matrix condition number is", condition_number(Q))

        print("The dimension is", dimension)

        print("Gradient descent part")
        x_gd, k_val, der_gd = gradient_descent(Q, c, dimension, thresholds, iterate_on_dim)
        check_solution(Q, c, x_gd[-1])

        print("Conjugate gradient part")
        x_cg, der_cg = conjugate_gradient(Q, c, dimension, thresholds, iterate_on_dim)
        check_solution(Q, c, x_cg[-1])

        plotting(k_val, der_gd, der_cg, download=True)

        print("---------------------------------")

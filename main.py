import numpy as np
import matplotlib.pyplot as plt


def plotting(k, der):
    plt.plot(k, der, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Derivative')
    plt.title('Graph of the derivative')
    plt.show()


def generate(dim):
    a_matrix = np.random.uniform(-1, 1, (dim, dim))
    c_vector = np.random.uniform(-1, 1, dim)
    return a_matrix, c_vector


def f(x, Q, c):
    return 0.5*np.dot(np.transpose(x), np.dot(Q, x)) + np.dot(np.transpose(c), x)


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


def gradient_descent(Q, c, dim):
    x = np.zeros(dim)
    history_x = [np.copy(x)]
    k_values = []
    der_values = []
    k = 0
    while k < dim:
        dk = direction(Q, x, c)
        ak = step_GD(Q, x, c)
        x = x + np.dot(ak, dk)
        # to plot
        k_values.append(k)
        der_values.append(gradient(Q, x, c))

        history_x.append(x.copy())
        k += 1
    plotting(k_values, der_values)
    return history_x


def step_CG(gk, Q, dk):
    ak = (np.linalg.norm(gk)) / (np.dot(np.transpose(dk), np.dot(Q, dk)))
    return ak


def conjugate_gradient(Q, c, dim):
    x = np.zeros(dim)
    history_x = [np.copy(x)]
    gk = np.dot(Q, x) + c
    dk = - gk
    k = 0
    while k < dim:
        ak = step_CG(gk, Q, dk)
        x = x + np.dot(ak, dk)
        history_x.append(x.copy())
        gk_new = gk + np.dot(ak, np.dot(Q, dk))
        bk = (np.linalg.norm(gk_new)) / (np.linalg.norm(gk))
        dk = - gk_new + np.dot(bk, dk)
        gk = gk_new
        k += 1
    return history_x


if __name__ == '__main__':
    # possible dimensions of the problem
    dimensions = [2, 3, 5, 7, 10, 20, 40, 50, 80, 100]

    # thresholds = [math.pow(10, -1), math.pow(10, -2), math.pow(10, -3), math.pow(10, -5), math.pow(10, -7)]

    for _, dimension in enumerate(dimensions):
        A, c = generate(dimension)
        Q = np.dot(A, np.transpose(A))

        x_gd = gradient_descent(Q, c, dimension)
        #print(x_gd[-1])
        #x_cg = conjugate_gradient(Q, c, dimension)
        #print(x_cg[-1])
        #print(x_gd)
        #print()
        #print(x_cg)

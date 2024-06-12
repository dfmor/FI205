import numpy as np
import cvxpy as cp
from random import random
from pylab import plot, show, xlabel, ylabel, grid, scatter, legend

d = 2  # dimensão
convergence_value = 10 ** (-3)
N_initial = 10  # number of initial values
N_alphas = 15
max_iterations = 100
N_points = 200


def random_qubit():
    u = random()
    v = random()

    theta = np.arccos(2 * v - 1)
    phi = 2 * np.pi * u

    return np.array([[np.cos(theta / 2)], [np.exp(phi * 1j) * np.sin(theta / 2)]])


def conjugate_transpose(matrix):
    return matrix.conj().T


def outer_product(vector1, vector2):
    return vector1 @ conjugate_transpose(vector2)


def random_observable():
    psi = random_qubit()
    return 2 * outer_product(psi, psi) - np.identity(d)


def optimal_rho(G):
    eigvals, eigvecs = np.linalg.eig(G)
    largest_eigenvector = np.c_[eigvecs[:, np.argmax(eigvals)]]
    return outer_product(largest_eigenvector, largest_eigenvector)


def G(alpha, A0, A1, B0, B1):
    return (
        alpha * cp.kron(A0, np.identity(d))
        + cp.kron(A0, B0)
        + cp.kron(A1, B0)
        + cp.kron(A0, B1)
        - cp.kron(A1, B1)
    )


def initial_G(alpha, A0, A1, B0, B1):
    return (
        alpha * np.kron(A0, np.identity(d))
        + np.kron(A0, B0)
        + np.kron(A1, B0)
        + np.kron(A0, B1)
        - np.kron(A1, B1)
    )


def seesaw(A0, A1, B0, B1, alpha):

    rho = optimal_rho(initial_G(alpha, A0, A1, B0, B1))

    gap = 1

    iterations = 0

    while (gap > convergence_value) and (iterations <= max_iterations):

        variable_A0 = cp.Variable(shape=(d, d), hermitian=True)
        variable_A1 = cp.Variable(shape=(d, d), hermitian=True)

        constraints_1 = [
            variable_A0 + np.identity(d) >> 0,
            variable_A1 + np.identity(d) >> 0,
            np.identity(d) - variable_A0 >> 0,
            np.identity(d) - variable_A1 >> 0,
        ]

        problem_1 = cp.Problem(
            cp.Maximize(
                cp.real(cp.trace(rho @ G(alpha, variable_A0, variable_A1, B0, B1)))
            ),
            constraints_1,
        )
        solution_1 = problem_1.solve()

        A0 = variable_A0.value
        A1 = variable_A1.value

        variable_B0 = cp.Variable(shape=(d, d), hermitian=True)
        variable_B1 = cp.Variable(shape=(d, d), hermitian=True)

        constraints_2 = [
            variable_B0 + np.identity(d) >> 0,
            variable_B1 + np.identity(d) >> 0,
            np.identity(d) - variable_B0 >> 0,
            np.identity(d) - variable_B1 >> 0,
        ]

        problem_2 = cp.Problem(
            cp.Maximize(
                cp.real(cp.trace(rho @ G(alpha, A0, A1, variable_B0, variable_B1)))
            ),
            constraints_2,
        )
        solution_2 = problem_2.solve()

        B0 = variable_B0.value
        B1 = variable_B1.value

        rho = optimal_rho(initial_G(alpha, A0, A1, B0, B1))

        gap = np.abs(solution_2 - solution_1)
        iterations += 1

    return solution_2


def repeat_seesaw(alpha):

    max_value = 0

    for _ in range(N_initial):

        A0 = random_observable()
        A1 = random_observable()
        B0 = random_observable()
        B1 = random_observable()

        value = seesaw(A0, A1, B0, B1, alpha)

        print(value)

        if value > max_value:
            max_value = value

    return max_value


alpha_list = np.linspace(0, 1, N_alphas)
value_list = []

for alpha in alpha_list:
    print("")
    print(f"alpha = {alpha}")
    print("")
    value_list.append(repeat_seesaw(alpha))

analytic_values = []
analytic_alpha = np.linspace(0, 1, N_points)
    
for i in analytic_alpha:
    analytic_values.append(np.sqrt(8 + 2*(i**2)))    


scatter(alpha_list, value_list)
plot(alpha_list, value_list, label = "see-saw")
plot(analytic_alpha, analytic_values, label = "analytic")
legend()
xlabel("alpha")
ylabel("max_value")
grid(True)

show()

import numpy as np
import cvxpy as cp
from random import random
from pylab import plot

d = 2  # dimensão
convergence_value = 10 ** (-3)
N_initial = 10  # number of initial values
max_iterations = 100


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


def G(A0, A1, A2, B0, B1, B2):
    return (
        -cp.kron(A0, np.identity(d))
        - cp.kron(A1, np.identity(d))
        - cp.kron(np.identity(d), B0)
        - cp.kron(np.identity(d), B1)
        - cp.kron(A0, B0)
        - cp.kron(A1, B0)
        - cp.kron(A2, B0)
        - cp.kron(A0, B1)
        - cp.kron(A1, B1)
        + cp.kron(A2, B1)
        - cp.kron(A0, B2)
        + cp.kron(A1, B2)
    )


def initial_G(A0, A1, A2, B0, B1, B2):
    return (
        -np.kron(A0, np.identity(d))
        - np.kron(A1, np.identity(d))
        - np.kron(np.identity(d), B0)
        - np.kron(np.identity(d), B1)
        - np.kron(A0, B0)
        - np.kron(A1, B0)
        - np.kron(A2, B0)
        - np.kron(A0, B1)
        - np.kron(A1, B1)
        + np.kron(A2, B1)
        - np.kron(A0, B2)
        + np.kron(A1, B2)
    )
    
def seesaw(A0, A1, A2, B0, B1, B2):  

    rho = optimal_rho(initial_G(A0, A1, A2, B0, B1, B2))

    gap = 1
    
    iterations = 0
    
    while(gap > convergence_value) and (iterations <= max_iterations):    

        variable_A0 = cp.Variable(shape=(d, d), hermitian=True)
        variable_A1 = cp.Variable(shape=(d, d), hermitian=True)
        variable_A2 = cp.Variable(shape=(d, d), hermitian=True)

        constraints_1 = [
            variable_A0 + np.identity(d) >> 0,
            variable_A1 + np.identity(d) >> 0,
            variable_A2 + np.identity(d) >> 0,
            np.identity(d) - variable_A0 >> 0,
            np.identity(d) - variable_A1 >> 0,
            np.identity(d) - variable_A2 >> 0,
        ]

        problem_1 = cp.Problem(
            cp.Maximize(
                cp.real(
                    cp.trace(rho @ G(variable_A0, variable_A1, variable_A2, B0, B1, B2))
                )
            ),
            constraints_1,
        )

        solution_1 = problem_1.solve()

        A0 = variable_A0.value
        A1 = variable_A1.value
        A2 = variable_A2.value
        
        variable_B0 = cp.Variable(shape=(2, 2), hermitian=True)
        variable_B1 = cp.Variable(shape=(2, 2), hermitian=True)
        variable_B2 = cp.Variable(shape=(2, 2), hermitian=True)

        constraints_2 = [
            variable_B0 + np.identity(d) >> 0,
            variable_B1 + np.identity(d) >> 0,
            variable_B2 + np.identity(d) >> 0,
            np.identity(d) - variable_B0 >> 0,
            np.identity(d) - variable_B1 >> 0,
            np.identity(d) - variable_B2 >> 0,
        ]

        problem_2 = cp.Problem(
            cp.Maximize(
                cp.real(
                    cp.trace(rho @ G(A0, A1, A2, variable_B0, variable_B1, variable_B2))
                )
            ),
            constraints_2,
        )

        solution_2 = problem_2.solve()

        B0 = variable_B0.value
        B1 = variable_B1.value
        B2 = variable_B2.value

        rho = optimal_rho(initial_G(A0, A1, A2, B0, B1, B2))

        gap = np.abs(solution_2 - solution_1)
        iterations += 1

    return solution_2


def repeat_seesaw():

    max_value = 0

    for _ in range(N_initial):

        A0 = random_observable()
        A1 = random_observable()
        A2 = random_observable()

        B0 = random_observable()
        B1 = random_observable()
        B2 = random_observable()

        value = seesaw(A0, A1, A2, B0, B1, B2)

        if value > max_value:
            max_value = value

    return max_value

print(repeat_seesaw())
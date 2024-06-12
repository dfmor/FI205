import numpy as np
import cvxpy as cp
import pylab as pl


def p_L(a, b, x, y):

    if (a == 0) and (b == 0):
        return 1

    else:
        return 0


def p_I(a, b, x, y):
    return 1 / 4


def p_PR(a, b, x, y):

    if ((a + b) % 2) == x * y:
        return 1 / 2

    else:
        return 0


def p(alpha, beta, a, b, x, y):
    return alpha * p_PR(a, b, x, y) + (1 - alpha) * (
        beta * p_L(a, b, x, y) + (1 - beta) * p_I(a, b, x, y)
    )


def pA(alpha, beta, a, x):

    soma = 0

    for b in [0, 1]:
        soma += p(alpha, beta, a, b, x, 0)

    return soma


def pB(alpha, beta, b, y):

    soma = 0

    for a in [0, 1]:
        soma += p(alpha, beta, a, b, 0, y)

    return soma

def matrix_element(matrix, i, j):

    vector_1 = np.eye(1, 5, i)[0]
    vector_2 = np.eye(1, 5, j).T

    return (vector_1 @ (matrix @ vector_2))[0]


def max_alpha(beta):

    gamma = cp.Variable(shape=(5, 5), hermitian=True)
    alpha = cp.Variable(nonneg=True)
    
    pA_00 = pA(alpha, beta, 0, 0)
    pA_01 = pA(alpha, beta, 0, 1)
    
    pB_00 = pB(alpha, beta, 0, 0)
    pB_01 = pB(alpha, beta, 0, 1)

    constraints = [
        alpha <= 1,
        gamma >> 0,
        matrix_element(gamma, 0, 0) == 1,
        matrix_element(gamma, 0, 1) == pA_00,
        matrix_element(gamma, 0, 2) == pA_01,
        matrix_element(gamma, 0, 3) == pB_00,
        matrix_element(gamma, 0, 4) == pB_01,
        matrix_element(gamma, 1, 1) == pA_00,
        matrix_element(gamma, 1, 3) == p(alpha, beta, 0, 0, 0, 0),
        matrix_element(gamma, 1, 4) == p(alpha, beta, 0, 0, 0, 1),
        matrix_element(gamma, 2, 2) == pA_01,
        matrix_element(gamma, 2, 3) == p(alpha, beta, 0, 0, 1, 0),
        matrix_element(gamma, 2, 4) == p(alpha, beta, 0, 0, 1, 1),
        matrix_element(gamma, 3, 3) == pB_00,
        matrix_element(gamma, 4, 4) == pB_01,               
    ]
    
    problem = cp.Problem(cp.Maximize(alpha), constraints)
    solution = problem.solve()
    
    return solution

lista_beta = np.linspace(0, 1, 50)

lista_alpha_quantum = []
for beta in lista_beta:
    lista_alpha_quantum.append(max_alpha(beta))

lista_alpha_local = []

for beta in lista_beta:
    
    alpha = cp.Variable(shape=(1,1))

    constraints = [alpha >= 0, alpha <= 1, 2*beta + alpha*(4 - 2*beta) <= 2]
    objective = cp.Maximize(alpha)

    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
   
    lista_alpha_local.append(alpha.value[0][0])
    
pl.plot(lista_beta, lista_alpha_quantum, label = "Q1")
pl.plot(lista_beta, lista_alpha_local, label = "local")

pl.xlabel("beta")
pl.ylabel("max_alpha")

pl.legend()
pl.grid()

pl.show()
    
    
    
    
    

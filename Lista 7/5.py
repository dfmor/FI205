import cvxpy as cp
import numpy as np
import pylab as pl

lista_beta = np.linspace(0,1,50)
lista_alpha = []

for beta in lista_beta:
    
    alpha = cp.Variable(shape=(1,1), name="alpha")

    constraints = [alpha >= 0, alpha <= 1, 2*beta + alpha*(4 - 2*beta) <= 2]
    objective = cp.Minimize(alpha)

    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
   
    lista_alpha.append(alpha.value[0][0])
    
pl.plot(lista_beta,lista_alpha)
pl.show()

    
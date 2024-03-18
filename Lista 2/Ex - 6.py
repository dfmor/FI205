import cvxpy as cp 
import numpy as np

dim = 4

A = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
B = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

X = cp.Variable((dim, dim), PSD = True) # Matrix dim x dim positiva semidefinida

constraints = [cp.trace(X) == 1, cp.trace(B @ X) == 1]

prob = cp.Problem(cp.Maximize(cp.trace(A @ X)), constraints)

solution = prob.solve()

print(solution)
print(X.value)
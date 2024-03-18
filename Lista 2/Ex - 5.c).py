import cvxpy as cp
import numpy as np

x = cp.Variable(shape=(2,1), name="x")

A = np.array([[-2,-1],[-1,-3],[-1,0],[0,-1]])
B = np.array([[-1],[-1],[0],[0]])

constraints = [cp.matmul(A, x) <= B]

r = np.array([1,0])
objective = cp.Minimize(cp.matmul(r, x))

problem = cp.Problem(objective, constraints)
solution = problem.solve()

print(solution)
print(x.value)
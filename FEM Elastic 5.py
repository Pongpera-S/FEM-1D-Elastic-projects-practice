import sys
import numpy as np
import matplotlib.pyplot as plt

# Finite element analysis for an elastic bar

# Problem definition
L = float(input("Enter length of the bar : "))  # length of the bar
E = float(input("Enter elastic modulus : "))  # elastic modulus
W = float(input("Enter weight of the bar : "))  # body force (weight of the bar)
w1 = float(input("Enter bar wiidth (top) : "))  # bar width (top)
w2 = float(input("Enter bar width (bottom) : "))  # bar width (bottom)
t = float(input("Enter bar thickness : "))  # bar thickness
P0 = float(input("Enter surface force (point load) at bottom end : "))  # surface force (point load) at bottom end

# Discretization
print('Discretize the problem domain')
e = int(input("Enter number of elements : "))  # number of elements
n = e + 1  # number of nodes

# node coordinates
print("Node coordinates")
x = []
print("Enter the entries for node coordinates:")
for i in range(n):
    x.append(float(input()))
x = np.array(x)
print(x)

# local destination array
print("Local destination array")
R = int(input("Enter the number of rows:"))
lda = []
print("Enter the entries for the local destination array (node indices of each element):")
for i in range(R):
    lda.append(list(map(int, input().split())))
lda = np.array(lda)
lda = lda-1
print(lda)

neb = np.array([int(input("Enter the essential boundary condition node:")) - 1])  # essential boundary condition node
nnb = np.array([int(input("Enter the natural boundary condition node:")) - 1])  # natural boundary condition node

# Element stiffness matrix and load vector
print('Create element stiffness matrix and load vector')
h = np.zeros(e)
A = np.zeros(e)
k = np.zeros(e)
f = np.zeros(e)
ke = np.zeros((2, 2, e))
fe = np.zeros((2, 1, e))

for i in range(e):
    h[i] = x[lda[1, i]] - x[lda[0, i]]  # length of element
    A[i] = t * (w1 - (w1 - w2) / L * (x[lda[1, i]] + x[lda[0, i]]) / 2)  # Average cross-sectional area
    k[i] = E * A[i] / h[i]
    f[i] = W * A[0] * h[i] / 2
    ke[:, :, i] = k[i] * np.array([[1, -1], [-1, 1]])
    fe[:, :, i] = f[i] * np.array([[1], [1]])
    print('Element stiffness matrix of element', i + 1, ':')
    print(ke[:, :, i])
    print('Element load vector for the body force of element', i + 1, ':')
    print(fe[:, :, i])

# Assemble the global stiffness matrix and load vector for the body force
K = np.zeros((n, n))
f_global = np.zeros((n, 1))

for i in range(e):
    K[lda[0, i], lda[0, i]] += ke[0, 0, i]
    K[lda[0, i], lda[1, i]] += ke[0, 1, i]
    K[lda[1, i], lda[0, i]] += ke[1, 0, i]
    K[lda[1, i], lda[1, i]] += ke[1, 1, i]

    f_global[lda[0, i], 0] += fe[0, 0, i]
    f_global[lda[1, i], 0] += fe[1, 0, i]

print('Global stiffness matrix:')
print(K)
print('Load vector for the body force:')
print(f_global)

# Apply the natural boundary condition
F = np.zeros((n, 1))
for i in range(len(nnb)):
    F[nnb[i], 0] = P0
print('Load vector for the surface force:')
print(F)

# Apply the essential boundary condition
for i in range(len(neb)):
    K[neb[i], :] = 0
    K[neb[i], neb[i]] = 1
    f_global[neb[i], 0] = 0
    F[neb[i], 0] = 0

print('Modified global stiffness matrix and load vectors:')
print(K)
print(f_global)
print(F)

# Solve for the nodal displacements
print('Displacements at nodes (nodal displacements):')
u = np.linalg.solve(K, f_global + F)
print(u)

# Calculate stresses in each element
sigma = np.zeros((e, 1))
for i in range(e):
    sigma[i, 0] = E * (u[lda[1, i], 0] - u[lda[0, i], 0]) / (x[lda[1, i]] - x[lda[0, i]])
print('Stress in each element:')
print(sigma)

# Analytical solution
xa = np.arange(0, L + 0.1, 0.1)
ua = (31 * np.log(20)) / 4160 - (31 * np.log(np.abs(xa - 20))) / 4160 + xa / 41600  # Displacement
sa = 250 - 77500 / (xa - 20)  # Stress

# Finite element solution
xe = np.empty(0)
ue = np.empty(0)
se = np.empty(0)
for i in range(e):
    xe = np.append(xe, [x[lda[0, i]], x[lda[1, i]]])
    ue = np.append(ue, [u[lda[0, i], 0], u[lda[1, i], 0]])
    se = np.append(se, [sigma[i, 0], sigma[i, 0]])

# Plot finite element solution vs. analytical solution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(xa, ua, 'r-', label='Analytical')
plt.plot(xe, ue, 'b+-', label='FEM')
plt.xlabel('x (in)')
plt.ylabel('Displacement (in)')

plt.subplot(1, 2, 2)
plt.plot(xa, sa, 'r-', label='Analytical')
plt.plot(xe, se, 'b+-', label='FEM')
plt.xlabel('x (in)')
plt.ylabel('Axial stress (psi)')

plt.tight_layout()
plt.show()
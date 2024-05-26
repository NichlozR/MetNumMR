import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, solve



x =     [0.0, 0.02, 0.036, 0.06, 0.094, 0.133, 0.164, 0.196, 0.234, 0.264, 0.285, 0.3]
y_data= [0.0, 4.7,  7.7,   10.5, 11.5,  10.0,  7.0,   6.0,   8.0,   12.0,  16.0,  19.0]

A = np.zeros((3*(len(x)-1), 3*(len(x)-1)))

last_pos = [0,0]


def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.copy(A)
    P = np.eye(n)

    for i in range(n):
        max_index = np.argmax(np.abs(U[i:n, i])) + i
        if i != max_index:
            U[[i, max_index]] = U[[max_index, i]]
            P[[i, max_index]] = P[[max_index, i]]
            if i > 0:
                L[[i, max_index], :i] = L[[max_index, i], :i]

        L[i, i] = 1

        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return P, L, U


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def invert_matrix(A):
    n = A.shape[0]
    P, L, U = lu_decomposition(A)
    inv_A = np.zeros((n, n))

    I = np.eye(n)
    for i in range(n):
        e = I[:, i]
        y = forward_substitution(L, np.dot(P, e))
        inv_A[:, i] = backward_substitution(U, y)

    return inv_A
def evaluate1(x):
    return x*x, x, 1

def exaluate2(x1, x2): ## x1 = xi-1, x2 = xi
    return 2*x1, 1, -2*x1, -1

for i in range(len(x)):
    #print(evaluate1(x[i]))
    if (i == 0):
        A[i][i], A[i][i + 1], A[i][i + 2] = evaluate1(x[i])
        last_pos = [0, 2]

    elif i == (len(x)-1):
        A[last_pos[0] + 1][last_pos[1] - 2], A[last_pos[0] + 1][last_pos[1] - 1], A[last_pos[0] + 1][last_pos[1]] = evaluate1(x[i])
        last_pos[0]+=1
        pass

    else:
        A[last_pos[0]+1][last_pos[1]-2], A[last_pos[0]+1][last_pos[1]-1], A[last_pos[0]+1][last_pos[1]] = evaluate1(x[i])
        A[last_pos[0]+2][last_pos[1] + 1], A[last_pos[0]+2][last_pos[1]+2], A[last_pos[0]+2][last_pos[1]+3] = evaluate1(x[i])
        last_pos[0]+=2
        last_pos[1]+=3

#pozycja startowa do etapu drugiego
last_pos[0]+=1
last_pos[1]=0

for i in range(2, len(x)):
    A[last_pos[0]][last_pos[1]], A[last_pos[0]][last_pos[1]+1], A[last_pos[0]][last_pos[1]+3], A[last_pos[0]][last_pos[1]+4] = exaluate2(x[i-1], x[i])
    last_pos[0]+=1
    last_pos[1]+=3

#etap 3
last_pos[1]=0
A[last_pos[0]][last_pos[1]] = 1

rows = 3*(len(x)-1)
y = np.zeros((rows, 1))
last_index = 0
for i in range(len(y_data)):
    if (last_index == 0):
        y[i][0] = y_data[i]
        last_index+=1
    elif (i == len(y_data)-1):
        y[last_index][0] = y_data[i]
    else:
        y[last_index][0] = y_data[i]
        y[last_index+1][0] = y_data[i]
        last_index+=2
print(y)

Ainv = invert_matrix(A)
#solutions = (Ainv.dot(y))

#print(A.dot(Ainv))

#print(solutions)

solutions = Ainv.dot(y)

print(solutions)

x_lines = []
y_lines = []

for i in range(len(x)-1):
    print(i)
    x_line = np.linspace(x[i], x[i+1], 1000)
    x_lines.append(x_line)
    #print(solutions[3*i][0],solutions[3*i + 1][0],solutions[3*i + 2][0])
    y_lines.append(solutions[3*i][0]*x_line*x_line + solutions[3*i + 1][0]*x_line + solutions[3*i + 2][0])

for i in range(len(x_lines)):
    plt.plot(x_lines[i], y_lines[i])

plt.scatter(x, y_data)

plt.xlabel("Voltage [V]")
plt.ylabel("Current [mA]")

plt.title('Square spline')

plt.savefig("square.png")
plt.show()
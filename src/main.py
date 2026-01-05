#importing relavent packages.
import numpy as np
import matplotlib.pyplot as plt
#inital condition.
#setting constants.
K = 1
l = 1
a = 0.4
b = 0.6
T = 1

#Compute Fourier Coefficient for initial condition (a).
def Bn1(n, a, b, T, l):
  return (2 * T / (n * np.pi)) * (np.cos((n * np.pi * a)/l) - np.cos((n * np.pi * b)/l))

#Defining the general solution.
def u_xt1(x, t, N, a, b, T,K):
  running_total = 0
  for n in range(1, N + 1):
    Bn = Bn1(n, a, b, T, l)
    running_total += (Bn) * np.sin((n * np.pi * x)/l) * np.exp(-K * (n * np.pi)**2 * t)
  return running_total
#Task 1 part 2 plots for u(x,0).
#defining relevant arrays.
x_vals = np.linspace(0, 1, 500)
Ns = [1, 10,100,250]
#plotting.
plt.figure(figsize=(10, 6))
for N in Ns:
    u_vals = [u_xt1(x, 0, N, a, b, T,K) for x in x_vals]
    plt.plot(x_vals, u_vals, label=f'N = {N}')

f_vals = [T if a <= x <= b else 0 for x in x_vals]
plt.plot(x_vals, f_vals, 'k--', label='f(x) (true)', linewidth=2)

plt.title('Fourier Approximation To f(x) For Various N And Inital Condition (a)')
plt.xlabel('x')
plt.ylabel('u(x, 0)')
plt.legend()
plt.grid(True)
plt.show()


#Compute Fourier Coefficient for initial condition (b).
def Bn2(n, a, b, T, l):
  return (32 * T / (((2*n-1) * (np.pi))**3))

#Defining the general solution.
def u_xt2(x, t, N, a, b, T,K):
  running_total = 0
  for n in range(1, N + 1):
    Bn = Bn2(n, a, b, T, l)
    running_total += (Bn) * np.sin(((2*n-1) * np.pi * x)/l) * np.exp(-K * ((2*n-1) * np.pi)**2 * t)
  return running_total
#Task 1 part 2 plots for u(x,0).
#defining relevant arrays.
x_vals = np.linspace(0, 1, 500)
Ns = [1, 10,100,500]
#plotting of figure.
plt.figure(figsize=(10, 6))
styles = ['-', '--', '-.', ':']
colors = ['blue', 'orange', 'green', 'red']

for i, N in enumerate(Ns):
    u_vals = [u_xt2(x, 0, N, a, b, T, K) for x in x_vals]
    plt.plot(x_vals, u_vals, linestyle=styles[i], color=colors[i], label=f'N = {N}')


f_vals = [(4*T*x*(l-x))/(l**2) for x in x_vals]
plt.plot(x_vals, f_vals, 'k--', label='f(x) (true)', linewidth=2)

plt.title('Fourier Approximation To f(x) For Various N And Inital Condition (b)')
plt.xlabel('x')
plt.ylabel('u(x, 0)')
plt.legend()
plt.grid(True)
plt.show()

#defining maximum N.
N = 250
#We make use of functions previously defined.
#defining arrays for position and time.
x_vals = np.linspace(0, 1, 500)
t_vals = [0, 0.002, 0.005, 0.01, 0.05, 0.1]

#plotting figure.
plt.figure(figsize=(10, 6))
for t in t_vals:
    u_vals = [u_xt1(x, t, N, a, b, T,K) for x in x_vals]
    plt.plot(x_vals, u_vals, label=f't = {t}')

plt.title(f'Time Evolution Of u(x, t) For N = {N} And Inital Condition (a)')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.show()

#defining maximum N.
N = 250
#We make use of functions previously defined.
#defining arrays for position and time.
x_vals = np.linspace(0, 1, 500)
t_vals = [0, 0.002, 0.005, 0.01, 0.05, 0.1]

#plotting figure.
plt.figure(figsize=(10, 6))
for t in t_vals:
    u_vals = [u_xt2(x, t, N, a, b, T,K) for x in x_vals]
    plt.plot(x_vals, u_vals, label=f't = {t}')

plt.title(f'Time Evolution Of u(x, t) For N = {N} And Inital Condition (b)')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.show()

#defining constants.
K = 1
l = 1
h = 0.01
epsilon = 1e-5
ns = int(l / h)
alpha = K * epsilon / h**2
beta = 1 - 2 * alpha
steps =  10000

# Grid
x = np.linspace(0, l, ns + 1)

# Initial condition: f(x) = 4x(1 - x)
u = 4 * x * (1 - x)

# Construct matrix A
A = np.zeros((ns + 1, ns + 1))
for i in range(1, ns):
    A[i, i - 1] = alpha
    A[i, i] = beta
    A[i, i + 1] = alpha
# A[0, :] and A[ns, :] remain zero (boundary conditions),

# Time stepping,

u_t = np.copy(u)
for k in range(steps):
    u_t = np.dot(A, u_t)  # Matrix multiplication,



# Plot.
plt.figure(figsize=(10, 6))
plt.plot(x, 4 * x * (1 - x), 'k--', label='Initial: f(x) = 4x(1-x)')
plt.plot(x, u_t, label=f'u(x, t={0.1})')
plt.title('Numerical Solution Of Heat Equation For Inital Condition (b)')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.show()

# Parameters
K = 1           # thermal diffusivity
l = 1           # length of the rod
h = 0.01        # spatial step size
epsilon = 1e-5  # time step size
ns = int(l / h) # number of spatial intervals
alpha = K * epsilon / h**2
beta = 1 - 2 * alpha
total_steps = int(0.1 / epsilon)  # simulate up to t = 0.1

# Spatial grid
x = np.linspace(0, l, ns + 1)

# Initial condition: Gaussian profile centered at 0.5
u = np.exp(-100 * (x - 0.5)**2)

# Construct the finite-difference matrix A
A = np.zeros((ns + 1, ns + 1))
for i in range(1, ns):
    A[i, i - 1] = alpha
    A[i, i]     = beta
    A[i, i + 1] = alpha
# First and last rows remain zero (Dirichlet boundary conditions)

# Snapshot times and corresponding step indices
snapshot_times = [0, 0.002, 0.005, 0.01, 0.05, 0.1]
snapshot_steps = [int(t / epsilon) for t in snapshot_times[1:]]  # skip t = 0 for loop
snapshots = [np.copy(u)]  # manually store initial condition (t = 0)

# Time-stepping loop
u_t = np.copy(u)
for k in range(1, total_steps + 1):
    u_t = np.dot(A, u_t)
    if k in snapshot_steps:
        snapshots.append(np.copy(u_t))

# Plot results
plt.figure(figsize=(10, 6))
for i, t in enumerate(snapshot_times):
    plt.plot(x, snapshots[i], label=f't = {t}')
plt.plot(x, u, 'k--', label='Initial condition', linewidth=2)

plt.title('Numerical Solution Of Heat Equation For Inital Condition (c)')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.show()

K = 1           # thermal diffusivity
l = 1           # length of the rod
h = 0.01        # spatial step size
epsilon = 1e-5  # time step size
ns = int(l / h) # number of spatial intervals
alpha = K * epsilon / h**2
beta = 1 - 2 * alpha
total_steps = int(0.1 / epsilon)  # simulate up to t = 0.1

# Spatial grid
x = np.linspace(0, l, ns + 1)

#n is manually set to 1 through 4.
n=4

# Initial condition: Gaussian profile centered at 0.5
u = np.sin(n*(np.pi * x)/l)

# Construct the finite-difference matrix A
A = np.zeros((ns + 1, ns + 1))
for i in range(1, ns):
    A[i, i - 1] = alpha
    A[i, i]     = beta
    A[i, i + 1] = alpha
# First and last rows remain zero (Dirichlet boundary conditions)

# Snapshot times and corresponding step indices
snapshot_times = [0, 0.002, 0.005, 0.01, 0.05, 0.1]
snapshot_steps = [int(t / epsilon) for t in snapshot_times[1:]]  # skip t = 0 for loop
snapshots = [np.copy(u)]  # manually store initial condition (t = 0)

# Time-stepping loop
u_t = np.copy(u)
for k in range(1, total_steps + 1):
    u_t = np.dot(A, u_t)
    if k in snapshot_steps:
        snapshots.append(np.copy(u_t))

# Plot results
plt.figure(figsize=(10, 6))
for i, t in enumerate(snapshot_times):
    plt.plot(x, snapshots[i], label=f't = {t}')
plt.plot(x, u, 'k--', label='Initial condition', linewidth=2)

plt.title('Numerical Solution Of Heat Equation For Inital Condition (d) m =4')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.show()

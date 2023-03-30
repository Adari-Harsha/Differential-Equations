import numpy as np
import matplotlib.pyplot as plt

#   Van der Pol equation
def f(t, y):
    u = y[0]
    v = y[1]
    return np.array([v, 40*(1 - u**2)*v - u])

# Defining the runge kutta method for the 4th order 
def rk4_method(ti, yi, dt, f):
    k1 = f(ti, yi)
    k2 = f(ti + dt/2, yi + k1*dt/2)
    k3 = f(ti + dt/2, yi + k2*dt/2)
    k4 = f(ti + dt, yi + k3*dt)
    return yi + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# Initial conditions for our van der pol equation
u0 = 0.1
v0 = 0
y0 = np.array([u0, v0])
t0 = 0
tf = 100
h = 0.001
time = np.arange(t0, tf, h)

# Solve the Van der Pol equation using the RK4 method
y = y0
u = [u0]
v = [v0]
for t in time:
    y = rk4_method(t, y, h, f)
    u.append(y[0])
    v.append(y[1])

# Plot the solution
plt.plot(time, u[:-1])
plt.xlabel('Time')
plt.ylabel('u(t)')
plt.title('Van der Pol equation')
plt.show()

plt.plot(u,v,'orange')
plt.xlabel('u(t)')
plt.ylabel('v(t)')
plt.title('Phase Diagaram')
plt.show()
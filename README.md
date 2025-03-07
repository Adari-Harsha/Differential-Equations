# Differential Equations

*   **Methods Implemented:**

This repository contains implementations of various numerical methods for solving ordinary differential equations (ODEs), including the Van der Pol equation.

## Introduction

This project provides a comprehensive overview of numerical methods used to solve ODEs, along with examples and visualizations.

## Methods Implemented

### 1. Explicit Euler's Method

The Butcher array for the Explicit Euler's Method is:
                                                       
                                                       ┌───┬───┐
                                                       │ 0 │ 0 │
                                                       ├───┼───┤
                                                       │   │ 1 │                                                       
                                                       └───┴───┘




The method is defined as:
- Ψ = f(t_i, u_i)
- u_(i+1) = u_i + hΨ = u_i + hf(t_i, u_i)

### 2. Improved Euler's Method

The Butcher array for the Improved Euler's Method is:


              ┌─────┬─────┬───┐
              │  0  │  0  │ 0 │
              │ 1/2 │ 1/2 │ 0 │
              ├─────┼─────┼───┤
              │     │  0  │ 1 │
              └─────┴─────┴───┘


Terms:
- K_1 = f(t, u)
- K_2 = f((t + 0.5h), (u + 0.5hK_1))
- Ψ = K_2
- Iteration formula: u_(i+1) = u_i + hf((t_i + 0.5h), (u_i + 0.5hf(t_i, u_i)))

### 3. Heun's Method

The Butcher array for Heun's Method is:

     ┌─────┬─────┬─────┬───┐
     │  0  │  0  │  0  │ 0 │
     │ 1/3 │ 1/3 │  0  │ 0 │
     │ 2/3 │  0  │ 2/3 │ 0 │
     ├─────┼─────┼─────┼───┤
     │     │ 1/4 │  0  │3/4│
     └─────┴─────┴─────┴───┘


Terms:
- K_1 = f(t_i, u_i)
- K_2 = f(t_i + h/3, u_i + (1/3)hK_1)
- K_3 = f(t_i + (2h/3), u_i + (2h/3)K_2)

### 4. Runge-Kutta (RK4) Method

Used for solving the Van der Pol equation.

## Example ODE and IVP

We use the following example ODE with Initial Value Problem (IVP):

u' = u + 2t, u(0) = 0

## Van der Pol Equation

The Van der Pol equation is a nonlinear ODE that models self-sustaining oscillations. It is given by:

$$ \frac{d^2x}{dt^2} - \mu(1-x^2)\frac{dx}{dt} + x = 0 $$

In this project, we solve it using the RK4 method.

### Van der Pol Equation Definition

The code defines a function `f(t, y)` that represents the Van der Pol equation, where `u = y[0]` and `v = y[1]`. The equation is expressed as:


### Runge-Kutta (RK4) Method

The code defines the Runge-Kutta method of the 4th order.


### Initial Conditions

The initial conditions are set as `u0 = 0.1` and `v0 = 0`, with `y0 = np.array([u0, v0])`. The start time is `t0 = 0`, and the end time is `tf = 100`, with a step size of `h = 0.001`.

### Solving the Equation

The Van der Pol equation is solved using the RK4 method in a loop that iterates through the time array.


### Plotting the Solution

The solution is plotted as `u(t)` vs. `Time`. A phase diagram is plotted as `u` vs. `v`.




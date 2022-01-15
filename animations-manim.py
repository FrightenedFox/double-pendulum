from scipy.integrate import odeint
import numpy as np
from numpy import sin, cos

# Parameters and constants
g = 9.8
θ1, θ2 = [0, 0]
ω1, ω2 = [4, 0]
m1, m2 = [1, 4]
L1, L2 = [5, 10]
μ1, μ2 = 0.1 * np.array([m1, m2])

u0 = (θ1, ω1)
u0_double = (θ1, θ2, ω1, ω2)
params = (g, L1, μ1)
params_double = (m1, m2, g, L1, L2)
params_double_wf = (m1, m2, g, L1, L2, μ1, μ2)

tspan = np.linspace(0.0, 100.0, 1001)


def single_pendulum_wf(u, tspan, g, L, μ):
    """Single pendulum with friction"""
    θ, ω = u
    dudt = [ω, -μ * ω - g / L * sin(θ)]
    return dudt


def double_pendulum(u, tspan, m1, m2, g, L1, L2):
    """Double pendulum without friction"""
    θ1, θ2, ω1, ω2 = u
    dudt = [
        ω1, 
        ω2,  
        ((-g*m2*(sin(2*θ2-θ1)-sin(θ1)) - L1*m2*(ω1**2)*sin(2*θ2-2*θ1) - 
        2*L2*m2*(ω2**2)*sin(θ2-θ1) + 2*g*m1*sin(θ1)) / 
        (L1*m2*(cos(2*(θ2-θ1))-1) - 2*L1*m1)),
        ((L2*m2*(ω2**2)*sin(2*θ2-2*θ1) + 2*L1*m2*(ω1**2)*sin(θ2-θ1) + 2*L1*m1*(ω1**2)*sin(θ2-θ1) + 
        g*m2*(sin(θ2-2*θ1)+sin(θ2)) + g*m1*(sin(θ2-2*θ1)+sin(θ2))) / 
        (L2*m2*(cos(2*(θ2-θ1))-1) - 2*L2*m1))
        ]
    return dudt


def double_pendulum_wf(u, tspan, m1, m2, g, L1, L2, μ1, μ2):
    """Double pendulum with friction"""
    θ1, θ2, ω1, ω2 = u
    dudt = [
        ω1, 
        ω2,  
        ((L1*ω1*(cos(2*(θ2-θ1))-1)*μ2 - 2*L1*ω1*μ1+g*m2*(sin(2*θ2-θ1)-sin(θ1)) + 
        L1*m2*(ω1**2)*sin(2*θ2-2*θ1) + 2*L2*m2*(ω2**2)*sin(θ2-θ1) - 2*g*m1*sin(θ1)) / 
        (2*L1*m1 - L1*m2*(cos(2*(θ2-θ1))-1))),
        ((L2*m2*ω2*(cos(2*(θ2-θ1))-1)*μ2 - 2*L1*m1*ω1*cos(θ2-θ1)*μ2 - 2*L2*m1*ω2*μ2 + 
        2*L1*m2*ω1*cos(θ2-θ1)*μ1 - L2*(m2**2)*(ω2**2)*sin(2*θ2-2*θ1) - 2*L1*(m2**2)*(ω1**2)*sin(θ2-θ1) -
        2*L1*m1*m2*(ω1**2)*sin(θ2-θ1) - g*(m2**2)*(sin(θ2-2*θ1)+sin(θ2)) - g*m1*m2*(sin(θ2-2*θ1)+sin(θ2))) / 
        (2*L2*m1*m2 - L2*(m2**2)*(cos(2*(θ2-θ1))-1)))
        ]
    return dudt


sol_single_wf = odeint(single_pendulum_wf, u0, tspan, args=params)
sol_double = odeint(double_pendulum, u0_double, tspan, args=params_double)
sol_double_wf = odeint(double_pendulum_wf, u0_double, tspan, args=params_double_wf)

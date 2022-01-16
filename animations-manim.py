import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos
from scipy.integrate import odeint

from manim import *


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


def prepare_quiver(bounds=(6*np.pi, 4*np.pi/3), steps=(0.2*np.pi, 0.2*np.pi), kh = 1):
    thetas = np.arange(-bounds[0], bounds[0], steps[0])
    omegas = np.arange(-bounds[1], bounds[1], steps[1])
    size = len(thetas) * len(omegas)
    thf, omf = np.empty(size, dtype=np.float64), np.empty(size, dtype=np.float64)
    slf = np.empty(size, dtype=np.float64)
    for i, θ in enumerate(thetas):
        for j, ω in enumerate(omegas):
            ind = i * len(omegas) + j
            thf[ind], omf[ind] = θ, ω
            slf[ind] = single_pendulum_wf((θ, ω), 0, *params)[1]
    return thf, omf, slf


thf, omf, slf = prepare_quiver()
# plt.plot(sol_single_wf[:, 0], sol_single_wf[:, 1], 'r', label="sol")
# plt.quiver(thf, omf,  omf, slf, color="b", alpha=0.5)
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.xlim(-1*np.pi/2, 11*np.pi/2)
# plt.ylim(-3*np.pi/3, 4*np.pi/3)
# plt.grid()
# plt.show()


class MyVectorField(Scene):
    def construct(self):
        # axes = NumberPlane([0, 12, 3], [-4, 4, 2])
        # self.add(axes)

        # length_func = lambda x: x / 6
        def func(pos):
            xy = pos
            # xy = axes.point_to_coords(pos)
            roc = single_pendulum_wf((xy[0], xy[1]), 0, *params)
            return roc[0] * LEFT + roc[1] * UP

        vf = ArrowVectorField(
            func, 
            x_range=[-10, 10, 0.5], 
            y_range=[-6, 6, 0.5], 
            opacity=0.9, 
            # length_func=length_func
            )
        self.add(vf)
        # self.wait()

        # dot = Dot().shift(LEFT)
        # vf.nudge(dot, -2, 60)
        # dot.add_updater(vf.get_nudge_updater())
        # theta_1_val = Text(f"pos: {dot.get_center()}").move_to(1.5 * DOWN)
        # update = lambda mob: mob.become(Text(f"pos: {dot.get_center()}").move_to(1.5 * DOWN))
        # theta_1_val.add_updater(update)

        # self.add(dot, theta_1_val)
        # self.wait(6)

        # plot = axes.plot_line_graph(
        #     sol_single_wf[:200, 0], 
        #     sol_single_wf[:200, 1], 
        #     vertex_dot_radius=0)

        # self.play(Create(plot, run_time=6))
        # self.wait()

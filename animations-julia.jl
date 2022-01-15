using DifferentialEquations
using Plots, LaTeXStrings
using Printf


function pendulum_problem!(du, u, params, t)
    μ, g, len = params
    θ, ω = u
    du[1] = ω
    du[2] = -μ * ω - g / len * sin(θ)
end

θ₀ = 0
ω₀ = pi
u₀ = [θ₀, ω₀]

μ = 0.1
g = 9.8
len = 5
params = [μ, g, len]

tspan = (0.0, 100.0)

prob = ODEProblem(pendulum_problem!, u₀, tspan, params)
sol = solve(prob, reltol=1e-5, saveat=10e-3)


function vec_to_list(vecvec, dim) 
    vecvec[dim]
end


@userplot pendulumanim
@recipe function fpend!(oa::pendulumanim)
    θ, title_time = oa.args
    title --> "Pendulum plot :: [$(@sprintf("%2.1f", title_time))%]"
    xaxis --> ("x", (-6,6))
    yaxis --> ("y", (-6,6))
    markersize --> 10
    markercolor --> "black"
    seriestype --> :scatter
    legend --> :none
    framestyle --> :grid
    aspect_ratio --> 1
    dpi --> 100
    [abs(len) * sin(θ)], [-abs(len) * cos(θ)]
end

function pendulum_example()

    # time frame
    t = 0:0.02:40
    n = length(t)
    k = 3

    # animation
    anim = @animate for i in 1:n
        plot([0, abs(len) * sin(sol.u[k*i][1])], [0, -abs(len) * cos(sol.u[k*i][1])])
        pendulumanim!(sol.u[k*i][1], i/n * 100)
    end

    gif(anim, "pendulum.gif", fps=50)

end

pendulum_example()
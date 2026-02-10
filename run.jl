include("linearised_modons.jl")

#U = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
#a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#β = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25]
#R = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#α = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#for i in 1:length(U)

#    run_modon(GPU(); U = U[i], a = a[i], β = β[i], R = R[i], α = α[i],
#              Nx = 1024, Ny = 1024, Lx = 10.24, Ly = 10.24, T = 200, Ns = 100, Nw = 1000,
#              save_fields = false, save_window = true)

#    run_modon(GPU(); U = U[i], a = a[i], β = β[i], R = R[i], α = α[i],
#              Nx = 2048, Ny = 2048, Lx = 20.48, Ly = 20.48, T = 500, Ns = 10, Nw = 2000,
#              save_fields = false, save_window = true)

#end

#μ = -0.9:0.1:1.0
#λ = 1.0

#for i in 1:length(μ)

#    run_modon(GPU(); μ = μ[i], λ, Nx = 2048, Ny = 2048, Lx = 20.48, Ly = 20.48,
#              T = 500, Ns = 10, Nw = 2000, save_fields = false, save_window = true)

#end

μ = -0.9:0.1:0
λ = 1.0

for i in 1:length(μ)

    run_modon(GPU(); μ = μ[i], λ, Nx = 2048, Ny = 2048, Lx = 20.48, Ly = 20.48, Wx = 6, Wy = 6,
              T = 80, Ns = 10, Nw = 400, save_fields = false, save_window = true, linear = true)

end

μ = 0.2:0.2:2.0
λ = 1.0

for i in 1:length(μ)

    run_modon(GPU(); μ = μ[i], λ, Nx = 2048, Ny = 2048, Lx = 20.48, Ly = 20.48, Wx = 6, Wy = 6,
              T = 1000, Ns = 10, Nw = 5000, save_fields = false, save_window = true, linear = true)

end

include("full_modon.jl")

U = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
β = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25]
R = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
α = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for i in 1:length(U)

    run_modon(GPU(); U = U[i], a = a[i], β = β[i], R = R[i], α = α[i],
              Nx = 1024, Ny = 1024, Lx = 10.24, Ly = 10.24, T = 200, Ns = 100, Nw = 1000,
              save_fields = false, save_window = true)

end
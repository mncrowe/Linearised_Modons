using GeophysicalFlows, NetCDF, Bessels, Roots, CUDA
using LinearAlgebra: mul!, ldiv!
using Random: seed!
seed!(1)

function run_modon(dev = GPU();
                     μ = nothing,	# nondimensional variable: βa²/U
                     λ = nothing,	# nondimensional variable: R/a
                     U = 1.0,		# modon speed in x-direction
                     a = 1.0,		# modon radius
                     β = 1.0,		# background PV gradient in y-direction
                     R = 1.0,		# (barotropic) Rossby radius of deformation
                     α = 0π/180,	# initial angle of modon
                    Nx = 1024,		# number of x gridpoints
                    Ny = 1024,		# number of y gridpoints
                    Lx = 10.24,		# length of x domain
                    Ly = 10.24,		# length of y domain
                     T = 200,		# total run time
                    Ns = 100,		# number of full field saves
                    Nw = 1000,		# number of window data saves
                    Wx = 8,		# window size in x direction (scalar or vector)
                    Wy = 4,		# window size in y direction (scalar or vector)
           save_fields = true,		# save full field true/false
           save_window = true,		# save window data true/false
)

    # set variables if nondimensional variables used:

    if !(μ isa Nothing)
        U, a, β = 1.0, 1.0, μ
    end

    if !(λ isa Nothing)
        R = a * λ
    end

    # Set filename for NetCDF data files:

    if μ isa Nothing
        savename = "data/U_" * sstring(U) * "_a_" * sstring(a) *
            "_beta_" * sstring(β) * "_R_" * sstring(R)
    else
        savename = "data/mu_" * sstring(β * a^2 / U) *
            "_lambda_" * sstring(R / a)
    end

    if !(α == 0)
        savename = savename * "_alpha_" * sstring(180*α/π)
    end

    # Define numerical parameters:

    stepper = "FilteredRK4"		          	# timestepping method, e.g. "RK4", "LSRK54" or "FilteredRK4"
    aliased_fraction = 0		            	# fraction of wavenumbers zeroed out in dealiasing
    κ₁, κ₂ = 2.5π, 5π					# range of wavenumbers to add random noise to
    r_c, σ = Lx/2-2, 0.5				# cutting parameters; radius, width of smoothing region
    nν = 2						# order of diffusivity/hyperdiffusivity
    ν = 0.0 * ((Lx/Nx)^2+(Ly/Ny)^2)^nν			# diffusivity/hyperdiffusivity
    Δt = 0.5 * ((Lx/Nx)+(Ly/Ny)) / (5*abs(U))		# timestep
    Nt = ceil(T / Δt)					# number of timesteps

    # Window save parameters:

    if length(Wx) == 1; Wx = [-Wx/2, Wx/2]; end		# set window range if width is specified
    if length(Wy) == 1; Wy = [-Wy/2, Wy/2]; end

    Lx₁, Lx₂ = Wx					# x range of save window
    Ly₁, Ly₂ = Wy					# y range of save window
    ix = Int.(Nx/2+1 .+ (Lx₁*Nx/Lx:Lx₂*Nx/Lx))		# x index for windowed save
    iy = Int.(Ny/2+1 .+ (Ly₁*Ny/Ly:Ly₂*Ny/Ly))		# y index for windowed save

    # Helper functions:

    to_dev(f) = device_array(dev)(f)
    
    function mod_dist(x, y, x_c, y_c)
        return @. sqrt((mod(x + Lx/2 - x_c, Lx) - Lx/2)^2 +
            (mod(y + Ly/2 - y_c, Ly) - Ly/2)^2)
    end

    function f_cut(x, y, x_c, y_c)
        r = mod_dist(x, y, x_c, y_c)
        mask = r .< r_c
        return @. mask + exp(-(r - r_c)^2 / σ^2) * !mask
    end

    # Create grid:

    grid = TwoDGrid(dev; nx=Nx, ny=Ny, Lx, Ly)
    x, y = gridpoints(grid)
    r, θ = to_CPU(sqrt.(x.^2 .+ y.^2)), to_CPU(atan.(y, x))

    # Define modon fields:

    p = sqrt(β/U + 1/R^2)

    J₁(x)  = besselj(1, x)
    K₁(x)  = besselk(1, x)
    J₁′(x) = (besselj(0, x) - besselj(2, x)) / 2
    K₁′(x) = (-besselk(0, x) - besselk(2, x)) / 2

    if p == 0

        K = 3.83170597020751231561443589 / a
        A = -U * a^2
        B = 2 * U / (K * J₁′(K * a))
    
        Ψᵢ = r -> B * J₁(K * r) - U * r
        Ψₒ = r -> A / r
        Qᵢ = r -> -K^2 * B * J₁(K * r)
        Qₒ = r -> 0

    else

        f(x) = x * J₁′(x) - (1 + x^2 / (p^2 * a^2)) * J₁(x) + x^2 * J₁(x) * K₁′(p * a) / (p * a * K₁(p * a))
        K′ = find_zero(f, 3.83170597020751231561443589)
        K = a * sqrt(K′^2 + 1/R^2)
    
        A = -U * a / K₁(p * a)
        B = p^2 * U * a / (K′^2 * J₁(K′ * a))
    
        Ψᵢ = r -> B * J₁(K′ * r) - U * (K′^2 + p^2) / K′^2 * r
        Ψₒ = r -> A * K₁(p * r)
        Qᵢ = r -> -K^2 / a^2 * B * J₁(K′ * r) + (U * p^2 * K^2 / (a^2 * K′^2) - β) * r;
        Qₒ = r -> β / U * A * K₁(p * r);

    end

    ψ = @. (Ψᵢ(r) * (r < a) + Ψₒ(r) * (r >= a)) * sin(θ - α)
    q = @. (Qᵢ(r) * (r < a) + Qₒ(r) * (r >= a)) * sin(θ - α)

    ψ[isnan.(ψ)] .= 0
    q[isnan.(q)] .= 0

    # Build problem:

    prob = SingleLayerQG.Problem(dev;
                                 nx=Nx,
                                 ny=Ny,
                                 Lx,
                                 Ly,
                                 β,
                                 U=-U,
                                 deformation_radius=R,
			         dt=Δt,
                                 stepper,
                                 aliased_fraction,
                                 ν,
                                 nν)

    # Create initial condition:

    κ = @.sqrt(prob.grid.Krsq)
    q₀ = to_dev(q) .+ 1e-5 * Nx * irfft(to_dev(exp.(im*2π*randn(Int(Nx/2+1), Int(Nx)))).*(κ.>κ₁).*(κ.<κ₂), Nx)

    SingleLayerQG.set_q!(prob, q₀)

    # Define output saves and do initial save:

    if save_fields

        filename = savename * ".nc"
        if isfile(filename); rm(filename); end

        nccreate(filename, "psi", "x", grid.x, "y", grid.y, "t", LinRange(0,T,Ns+1))
        nccreate(filename, "q", "x", grid.x, "y", grid.y, "t", LinRange(0,T,Ns+1))
        ncputatt(filename," ", Dict("R" => R, "U" => U, "a" => a, "beta" => β, "alpha" => α))

        save_field_data(prob, filename, 0)

    end

    if save_window

        windowname = savename * "_window.nc"
        if isfile(windowname); rm(windowname); end

        nccreate(windowname, "q", "x", grid.x[ix], "y", grid.y[iy], "t", LinRange(0,T,Nw+1))
        ncputatt(windowname," ", Dict("R" => R, "U" => U, "a" => a, "beta" => β, "alpha" => α))

        save_window_data(prob, windowname, 0, ix, iy)

    end

    # Print current (lack of) progress:

    print_output(prob, 0)

    # Run simulation:

    I = gcdx(Int(Nt/Ns), Int(Nt/Nw))[1]

    for i1 in 1:ceil(Nt/I)

        # Evolve problem forward in time:

        stepforward!(prob, I)
        SingleLayerQG.updatevars!(prob)

        if maximum(isnan.(prob.sol)); @warn "NaN detected."; end

        x_c, y_c = get_centre(prob, prob.grid)
        q₁ = prob.vars.q .* f_cut(x, y, x_c, y_c)     #(f_cut(x .- Lx, y, x_c, 0) .+ f_cut(x, y, x_c, 0) .+ f_cut(x .+ Lx, y, x_c, 0))
	SingleLayerQG.set_q!(prob, q₁)

        # Save full fields:

        if (i1*I % Int(Nt/Ns) == 0) & save_fields
            save_field_data(prob, filename, Int(i1*I*Ns/Nt))
        end

        # Save windowed data:

        if (i1*I % Int(Nt/Nw) == 0) & save_window
            save_window_data(prob, windowname, Int(i1*I*Nw/Nt), ix, iy)
        end

        # Print current iteration, simulation time, and maximum PV value:

        print_output(prob, i1*I)

    end

    return nothing

end

function save_field_data(problem, filename, i)

    grid = problem.grid
    Nx, Ny = grid.nx, grid.ny
    ψ, q = reshape(to_CPU(problem.vars.ψ),(Nx, Ny, 1)), reshape(to_CPU(problem.vars.q),(Nx, Ny, 1))

    ncwrite(ψ, filename, "psi", start = [1, 1, i+1], count = [Nx, Ny, 1])
    ncwrite(q, filename, "q", start = [1, 1, i+1], count = [Nx, Ny, 1])

    return nothing

end

function save_window_data(problem, filename, i, ix, iy)

    S_w = (length(ix), length(iy), 1)
    Nx, Ny = problem.grid.nx, problem.grid.ny

    Q = to_CPU(problem.vars.q)

    x_c, y_c = get_centre(problem, problem.grid)
    _, i_m = findmin((problem.grid.x .- x_c).^2)
    _, j_m = findmin((problem.grid.y .- y_c).^2)

    ix_c = Int.(mod.((-(Nx/2-1):(Nx/2)) .+ i_m .+ 1, Nx))
    ix_c[ix_c .== 0] .= Nx

    iy_c = Int.(mod.((-(Ny/2-1):(Ny/2)) .+ j_m .+ 1, Ny))
    iy_c[iy_c .== 0] .= Ny

    Q_s = reshape(Q[ix_c, iy_c][ix, iy], S_w)
    ncwrite(Q_s, filename, "q", start = [1, 1, i+1], count = collect(S_w))

    return nothing
end

function get_centre(problem, grid)

    Lx, Ly = grid.Lx, grid.Ly
    x, y = grid.x, grid.y

    _, imax = findmax(problem.vars.q)
    _, imin = findmin(problem.vars.q)

    x_c, y_c = (x[imax[1]] + x[imin[1]]) / 2, (y[imax[2]] + y[imin[2]]) / 2

    if abs(x[imax[1]] - x[imin[1]]) > Lx / 2
        x_c = x_c < 0 ? x_c + Lx/2 : x_c - Lx/2
    end

    if abs(y[imax[2]] - y[imin[2]]) > Ly / 2
        y_c = y_c < 0 ? y_c + Ly/2 : y_c - Ly/2
    end

    return x_c, y_c
end

function print_output(problem, iter)
    println("Iteration: " * istring(iter) * ", t = " * fstring(problem.clock.t) *
            ", Δq = " * sstring(abs(maximum(problem.vars.q) + minimum(problem.vars.q))))
    return nothing
end

# Helper functions:

to_CPU(f) = device_array(CPU())(f)
fstring(num) = string(round(num, sigdigits=8))
sstring(num) = string(round(num, sigdigits=2))
istring(num) = string(Int(num))

nothing
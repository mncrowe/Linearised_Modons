# Defines the linearised QG problem using FourierFlows.jl

using FourierFlows, CUDA
using LinearAlgebra: mul!, ldiv!

# Define parameter and variables structures:

struct ParamsFiniteR{T, Aphys} <: AbstractParams
   ν :: T
  nν :: Int
   β :: T
   R :: T
   U :: T
  Ψx :: Aphys
  Ψy :: Aphys
  Qx :: Aphys
  Qy :: Aphys
end

struct ParamsInfiniteR{T, Aphys} <: AbstractParams
   ν :: T
  nν :: Int
   β :: T
   U :: T
  Ψx :: Aphys
  Ψy :: Aphys
  Qx :: Aphys
  Qy :: Aphys
end

struct Vars{Aphys, Atrans} <: AbstractVars
   q :: Aphys
   ψ :: Aphys
   u :: Aphys
   v :: Aphys
  qh :: Atrans
  ψh :: Atrans
  uh :: Atrans
  vh :: Atrans
end

function Vars(grid)
  
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh ψh uh vh

  return Vars(q, ψ, u, v, qh, ψh, uh, vh)
end

# Define linear and nonlinear terms:

function calcN!(N, sol, t, clock, vars, params, grid)

  dealias!(sol, grid)

  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  uQx, vQy = vars.u, vars.v
  uQxh, vQyh = vars.uh, vars.vh

  @. uQx *= params.Qx
  @. vQy *= params.Qy

  mul!(uQxh, grid.rfftplan, uQx)
  mul!(vQyh, grid.rfftplan, vQy)
  
  @. N = - uQxh - vQyh

  qx, qy = vars.u, vars.v
  qxh, qyh = vars.uh, vars.vh

  @. qxh = im * grid.kr * vars.qh
  @. qyh = im * grid.l  * vars.qh

  ldiv!(qx, grid.rfftplan, qxh)
  ldiv!(qy, grid.rfftplan, qyh)

  JΨq, JΨqh = vars.ψ, vars.ψh

  @. JΨq = qx * params.Ψy - qy * params.Ψx

  mul!(JΨqh, grid.rfftplan, JΨq)
  
  @. N += JΨqh

  return nothing
end

function calcNnonlin!(N, sol, t, clock, vars, params, grid)

  calcN!(N, sol, t, clock, vars, params, grid)

  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.q, grid.rfftplan, vars.qh)
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  uq, vq = vars.u, vars.v                   # use vars.u, vars.v as scratch variable
  @. uq *= vars.q                           # uq
  @. vq *= vars.q                           # vq

  uqh, vqh = vars.uh, vars.vh               # use vars.uh, vars.vh as scratch variable
  mul!(uqh, grid.rfftplan, uq)              # \hat{uq}
  mul!(vqh, grid.rfftplan, vq)              # \hat{vq}

  @. N += -im * grid.kr * uqh - im * grid.l * vqh  # - ∂[uq]/∂x - ∂[vq]/∂y

  return nothing
end

function Equation(params::ParamsInfiniteR, grid, nonlinear)
 
  L = @. im * params.U * grid.kr - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr * grid.invKrsq

  CUDA.@allowscalar L[1, 1] = 0

  calcNfunc! = nonlinear ? calcNnonlin! : calcN!

  return FourierFlows.Equation(L, calcNfunc!, grid)
end

function Equation(params::ParamsFiniteR, grid, nonlinear)
 
  L = @. im * params.U * grid.kr - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr / (grid.Krsq + 1 / params.R^2) 

  CUDA.@allowscalar L[1, 1] = 0

  calcNfunc! = nonlinear ? calcNnonlin! : calcN!

  return FourierFlows.Equation(L, calcNfunc!, grid)
end

function set_q!(prob, q)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid

  mul!(vars.qh, grid.rfftplan, q)
  @. sol = vars.qh

  updatevars!(sol, vars, params, grid)

  return nothing
end

function updatevars!(sol, vars, params, grid)
  dealias!(sol, grid)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.q, grid.rfftplan, deepcopy(vars.qh))
  ldiv!(vars.ψ, grid.rfftplan, deepcopy(vars.ψh))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))

  return nothing
end

function streamfunctionfrompv!(ψh, qh, params::ParamsFiniteR, grid)
  @. ψh =  - qh / (grid.Krsq + 1 / params.R^2)
  return nothing
end

function streamfunctionfrompv!(ψh, qh, params::ParamsInfiniteR, grid)
  @. ψh =  - qh * grid.invKrsq
  return nothing
end

updatevars!(prob) = updatevars!(prob.sol, prob.vars, prob.params, prob.grid)

function LinQGProblem(dev::Device = CPU();
                        nonlinear = false,
                               nx = 256,
                               ny = 256,
                               Lx = 2π,
                               Ly = Lx,
                                β = 0.0,
               deformation_radius = Inf,
                                U = 0.0,
                               ψx = nothing,
                               ψy = nothing,
                               qx = nothing,
                               qy = nothing,
                                ν = 0.0,
                               nν = 1,
                               dt = 0.01,
                          stepper = "RK4",
                 aliased_fraction = 1/3,
                                T = Float64)

    grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)

    U = U isa Number ? convert(T, U) : U

    ψx isa Nothing && (ψx = zeros(dev, T, (nx, ny)))
    ψy isa Nothing && (ψy = zeros(dev, T, (nx, ny)))
    qx isa Nothing && (qx = zeros(dev, T, (nx, ny)))
    qy isa Nothing && (qy = zeros(dev, T, (nx, ny)))

    params = deformation_radius == Inf ? ParamsInfiniteR(ν, nν, β, U, ψx, ψy, qx, qy) :
                                         ParamsFiniteR(ν, nν, β, deformation_radius, U, ψx, ψy, qx, qy)

    vars = Vars(grid)
    equation = Equation(params, grid, nonlinear)

    return FourierFlows.Problem(equation, stepper, dt, grid, vars, params)

end
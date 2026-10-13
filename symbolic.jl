"""
    FPSoundSymbolic

Floating-point sound symbolic propagation (FPSoundSymbolic in the paper). Each neuron is
represented by an interval linear expression ([`IntervalAffExpr`](@ref IntervalExpr)) of the input
variables and of the relaxation variables of unstable ReLUs. In the affine layers each
addend expression is multiplied coefficient-wise by `[1 - Δ, 1 + Δ]` (Eq. (7) and
Proposition 3 of the paper, see [`safe_sum`](@ref)); ReLU uses the relaxation of Eq. (8).

Use [`FPSoundSymbolic.Algorithm`](@ref) with [`FPSoundSymbolic.Option`](@ref), normally
through [`run_on_sample`](@ref) or [`run_on_batch`](@ref).
"""
module FPSoundSymbolic
import ..FPSoundVerification as FPSV
using ..FPSoundVerification

using Flux
using IntervalArithmetic

"""
    FPSoundSymbolic.Option

Options of the verifier. Create them with `from_kwargs(FPSoundSymbolic.Option; ...)` or
`from_dict(FPSoundSymbolic.Option, dict)`.

- `input_range::Tuple`: valid input range `(lo, hi)`; the input box is clipped to it. Both
  elements must be of type `u_type`.
- `u_type::Type = Float64`: verified precision (`Float32` or `Float64`). It must equal the
  precision used in deployment. The network, the input and the radius are converted to it
  (see `assert_type!`), and all bounds are computed in it.
- `fp_sound::Bool = true`: account for floating-point rounding (flag `-s`). If `false`, the
  bounds hold for real arithmetic only and `over_safe` and `under_safe` have no effect.
- `backend::String = "textbook"`: only `"textbook"` is supported.
- `over_safe::Bool = true`: detect possible overflows (flag `-o`), see [`safe_sum`](@ref).
- `under_safe::Bool = false`: keep the bounds sound in flush-to-zero / denormals-are-zero
  environments (flag `-u`), see [`safe_sum`](@ref). Requires `over_safe`; if it is set
  without `over_safe`, both are used (with a warning).

See [`Decision`](@ref) for what the result guarantees under each combination.
"""
@option struct Option <: AbstractOptions
    input_range::Tuple
    u_type::Type = Float64
    fp_sound::Bool = true
    backend::String = "textbook"
    over_safe::Bool = true
    under_safe::Bool = false
end


"""
    FPSoundSymbolic.Algorithm{T}(network, input_point, true_index, perturbation_radius, option)

State of one symbolic verification run. `T` must equal `option.u_type`;
[`run_on_sample`](@ref) creates it this way.

- `network`, `input_point`, `true_index`, `perturbation_radius`: as in
  [`FPSoundIBP.Algorithm`](@ref).
- `vars::VariableManager`: domains of the input variables and of the ReLU relaxation
  variables.
- `prop_expr`: the current bounds, one `IntervalAffExpr{T}` per neuron of the current layer.
- `results::Dict{Symbol, Any}`: filled during the run (`:UnstableNodes`, `:UnderOverRisk`,
  `:OutputBounds`, `:HostileIndices`, `:Status`).
- `sum_fn`: neuron summation routine, selected by `init_bounding_algorithm!`.
"""
mutable struct Algorithm{T}
    solve_options::Option
    perturbation_radius::Real
    network::Flux.Chain
    flattened_network::Vector
    vars::VariableManager{Symbol, <:Real}

    input_point::AbstractArray{<:Real}
    prop_expr::AbstractArray{IntervalAffExpr{T}}

    true_index::Int

    results::Dict{Symbol, Any}

    sum_fn::Function

    function Algorithm{T}(network::Flux.Chain, input_point::AbstractArray{<:Real}, true_index::Int, perturbation_radius::Real, option::Option) where T
        new(
            option,
            perturbation_radius,
            network,
            [],
            VariableManager(0, option.u_type),
            input_point,
            [],
            true_index,
            Dict{Symbol, Any}(),
            x -> (sum(x), false)
        )
    end

end


"""
    register_var_and_create_input_expr!(input_domain, vm) -> IntervalAffExpr

Register a new variable with domain `input_domain` in `vm` and return the expression
`1 · x_new`.
"""
function register_var_and_create_input_expr!(input_domain::Interval{T}, vm::VariableManager{S, T}) where {S, T<:Real}
    new_var = register_new_var!(vm, input_domain)
    return IntervalExpr{S, T}(interval(T, 0), OrderedDict{S, Interval{T}}(new_var => interval(T, 1)))
end


"""
    init_bounding_algorithm!(algorithm::FPSoundSymbolic.Algorithm)

Convert the inputs to `u_type`, register one variable per input coordinate with its range
in the input box (see `get_input_domain`), set `prop_expr` to these variables,
split the network into elementary layers and reset `results[:UnderOverRisk]` and
`results[:UnstableNodes]`.

The summation routine `sum_fn` used for the affine layers is selected as follows:

| `fp_sound` | `over_safe` | `under_safe` | `sum_fn` |
|:--- |:--- |:--- |:--- |
| `false` | any | any | not used; plain outward-rounded sums (real arithmetic) |
| `true` | `false` | `false` | [`safe_sum_base`](@ref) |
| `true` | `true` | `false` | [`safe_sum`](@ref) |
| `true` | `true` | `true` | [`safe_sum`](@ref) with `under_safe = true` |
| `true` | `false` | `true` | as the previous row, with a warning |
"""
function FPSV.init_bounding_algorithm!(algorithm::Algorithm)
    @assert algorithm.solve_options.backend in ["textbook"] "Unsupported backend: $(algorithm.solve_options.backend), only 'textbook' is supported."

    assert_type!(algorithm)

    input_domains = get_input_domain(algorithm.input_point, algorithm.perturbation_radius, algorithm.solve_options.input_range)

    algorithm.prop_expr = register_var_and_create_input_expr!.(input_domains, Ref(algorithm.vars))

    algorithm.flattened_network = normalize_network_layers(algorithm.network)

    algorithm.results[:UnderOverRisk] = false
    algorithm.results[:UnstableNodes] = 0

    over_safe = algorithm.solve_options.over_safe
    if !over_safe && algorithm.solve_options.under_safe && algorithm.solve_options.fp_sound
        @warn "under_safe alone is not supported. Setting over_safe to true."
        over_safe = true
    end
    
    if over_safe && !algorithm.solve_options.under_safe && algorithm.solve_options.fp_sound
            algorithm.sum_fn = x -> safe_sum(x, algorithm.solve_options.u_type, algorithm.vars)
    elseif over_safe && algorithm.solve_options.under_safe && algorithm.solve_options.fp_sound
            algorithm.sum_fn = x -> safe_sum(x, algorithm.solve_options.u_type, algorithm.vars; under_safe=true)
    elseif !over_safe && !algorithm.solve_options.under_safe && algorithm.solve_options.fp_sound
            algorithm.sum_fn = x -> safe_sum_base(x, algorithm.solve_options.u_type, algorithm.vars)
    end


end


"""
    apply_relu_on_exp(exp, vars) -> (relu_exp, unstable::Bool)

Symbolic ReLU (Eq. (8) of the paper).
"""
function apply_relu_on_exp(exp::IntervalExpr{S, T}, vars::VariableManager{S, T}) where {S, T<:Real}
    conc_bounds = exp(vars)
    unstable = false

    if sup(conc_bounds) <= 0
        res = zero(exp)
    elseif  inf(conc_bounds) >= 0
        res = exp
    else
        iu = interval(sup(conc_bounds))
        il = interval(inf(conc_bounds))
        
        lambda = iu / (iu - il)
        
        res = exp * lambda

        M = lambda * interval(abs(inf(conc_bounds)))

        M_domain = interval(T, 0, sup(M))
        new_var_sym = get_next_var_sym(vars)

        extend_exp_with_new_var!(res, new_var_sym, interval(T, 1))
        register_new_var!(vars, M_domain)

        unstable = true
    end

    return (res, unstable)
end


function model_network_layer!(layer::DotFunctor{ReLUType}, algorithm::Algorithm)
    res = similar(algorithm.prop_expr)

    for i in eachindex(algorithm.prop_expr)
        res[i], unstable = apply_relu_on_exp(algorithm.prop_expr[i], algorithm.vars)
        algorithm.results[:UnstableNodes] += unstable
    end

    algorithm.prop_expr = res
end


function model_network_layer!(layer::Flux.Dense, algorithm::Algorithm)
    res, risk = apply_dense(algorithm.prop_expr, layer, algorithm.solve_options.u_type, algorithm.solve_options.fp_sound; implementation=algorithm.solve_options.backend, sum_fn=algorithm.sum_fn, vm = algorithm.vars)

    algorithm.prop_expr = res
    algorithm.results[:UnderOverRisk] = algorithm.results[:UnderOverRisk] || risk
end


function model_network_layer!(layer::Flux.Conv, algorithm::Algorithm)
    res, risk = apply_conv(algorithm.prop_expr, layer, algorithm.solve_options.u_type, algorithm.solve_options.fp_sound; implementation=algorithm.solve_options.backend, sum_fn=algorithm.sum_fn, vm = algorithm.vars)

    algorithm.prop_expr = res
    algorithm.results[:UnderOverRisk] = algorithm.results[:UnderOverRisk] || risk
end


function model_network_layer!(layer::FlattenType, algorithm::Algorithm)
    algorithm.prop_expr = Flux.flatten(algorithm.prop_expr)
end


function model_network_layer!(layer::NormalizeLayer, algorithm::Algorithm)
    algorithm.prop_expr = layer(algorithm.prop_expr)
end


"""
    model_network_layers!(algorithm::FPSoundSymbolic.Algorithm) -> Array{IntervalAffExpr{T}}

Propagate the expressions through every layer and return the output expressions:

- `Dense`, `Conv`: [`apply_dense`](@ref), [`apply_conv`](@ref) with `algorithm.sum_fn`; a
  reported possible overflow sets `results[:UnderOverRisk]`;
- `relu`: [`apply_relu_on_exp`](@ref) for each neuron; unstable neurons are counted in
  `results[:UnstableNodes]`;
- `Flux.flatten`, [`NormalizeLayer`](@ref): applied to the expressions.
"""
function FPSV.model_network_layers!(algorithm::Algorithm)
    for layer in algorithm.flattened_network
        model_network_layer!(layer, algorithm)

        # # Debug
        if layer isa Flux.Dense && length(layer.bias) == 101
            println(algorithm.prop_expr[end](algorithm.vars))
        end
    end

    return algorithm.prop_expr
end


concretize(x::IntervalAffExpr, vars::VariableManager) = x(vars)


"""
    decide_label_uniformity!(algorithm::FPSoundSymbolic.Algorithm)

Concretize each output expression over `algorithm.vars` and decide as
[`decide_label_uniformity!`](@ref) for `FPSoundIBP`: the intervals `o` are stored in
`results[:OutputBounds]`, and the status is computed from `sup(o_i - o_t)`. See
[`Decision`](@ref).
"""
function FPSV.decide_label_uniformity!(algorithm::Algorithm)
    out_bounds = concretize.(algorithm.prop_expr |> vec, Ref(algorithm.vars))

    algorithm.results[:OutputBounds] = out_bounds

    out_bounds = out_bounds .- out_bounds[algorithm.true_index]

    algorithm.results[:HostileIndices] = []
    
    if algorithm.results[:UnderOverRisk]
        algorithm.results[:Status] = PossibleUnderOrOverflow
        return
    end

    for i in eachindex(out_bounds)
        if i != algorithm.true_index && sup(out_bounds[i]) > 0
            algorithm.results[:Status] = Adversarial
            push!(algorithm.results[:HostileIndices], i)
        end
    end

    if isempty(algorithm.results[:HostileIndices])
        algorithm.results[:Status] = Safe
    end
    
    return
end

end # module FPSoundSymbolic
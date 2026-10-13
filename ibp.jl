"""
    FPSoundIBP

Floating-point sound interval bound propagation (FPSoundIBP in the paper). Each neuron is
bounded by an interval. In the affine layers the neuron sums are computed with
[`safe_sum`](@ref), so that the intervals contain every floating-point result of the
deployed network (Eq. (5) and Proposition 2 of the paper).

"""
module FPSoundIBP

import ..FPSoundVerification as FPSV
using ..FPSoundVerification

using Flux
using IntervalArithmetic

"""
    FPSoundIBP.Option

Options of the verifier. Create them with `from_kwargs(FPSoundIBP.Option; ...)` or
`from_dict(FPSoundIBP.Option, dict)`.

- `input_range::Tuple`: valid input range `(lo, hi)`; the input box is clipped to it. Both
  elements must be of type `u_type`.
- `u_type::Type = Float64`: verified precision (`Float32` or `Float64`). It must equal the
  precision used in deployment. The network, the input and the radius are converted to it
  (see `assert_type!`), and all bounds are computed in it.
- `fp_sound::Bool = true`: account for floating-point rounding (flag `-s`). If `false`, the
  bounds hold for real arithmetic only and `over_safe` and `under_safe` have no effect.
- `backend::String = "textbook"`: only `"textbook"` is supported now.
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
    FPSoundIBP.Algorithm{T}(network, input_point, true_index, perturbation_radius, option)

State of one IBP verification run. `T` must equal `option.u_type`; [`run_on_sample`](@ref)
creates it this way.

- `network::Flux.Chain`: the classifier.
- `input_point`: the unperturbed input.
- `true_index`: 1-based index of the correct class.
- `perturbation_radius`: radius of the L∞ ball around `input_point`.
- `prop_expr`: the current bounds, one `Interval{T}` per neuron of the current layer.
- `results::Dict{Symbol, Any}`: filled during the run (`:UnstableNodes`, `:UnderOverRisk`,
  `:OutputBounds`, `:HostileIndices`, `:Status`).
- `sum_fn`: neuron summation routine, selected by `init_bounding_algorithm!`.
"""
mutable struct Algorithm{T}
    solve_options::Option
    perturbation_radius::Real
    network::Flux.Chain
    flattened_network::Vector

    input_point::AbstractArray{<:Real}
    prop_expr::AbstractArray{Interval{T}}

    true_index::Int

    results::Dict{Symbol, Any}

    sum_fn::Function

    function Algorithm{T}(network::Flux.Chain, input_point::AbstractArray{<:Real}, true_index::Int, perturbation_radius::Real, option::Option) where T
        new(
            option,
            perturbation_radius,
            network,
            [],
            input_point,
            [],
            true_index,
            Dict{Symbol, Any}(),
            x -> (sum(x), false)
        )
    end

end

"""
    init_bounding_algorithm!(algorithm::FPSoundIBP.Algorithm)

Convert the inputs to `u_type`, set `prop_expr` to the input box (see
`get_input_domain`), split the network into elementary layers and reset
`results[:UnderOverRisk]` and `results[:UnstableNodes]`.

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

    algorithm.prop_expr = get_input_domain(algorithm.input_point, algorithm.perturbation_radius, algorithm.solve_options.input_range)
    # algorithm.prop_expr = convert.(Interval{Float64}, input_domains)

    algorithm.flattened_network = normalize_network_layers(algorithm.network)

    algorithm.results[:UnderOverRisk] = false
    algorithm.results[:UnstableNodes] = 0

    over_safe = algorithm.solve_options.over_safe
    if !over_safe && algorithm.solve_options.under_safe && algorithm.solve_options.fp_sound
        @warn "under_safe alone is not supported. Setting over_safe to true."
        over_safe = true
    end

    if over_safe && !algorithm.solve_options.under_safe && algorithm.solve_options.fp_sound
            algorithm.sum_fn = x -> safe_sum(x, algorithm.solve_options.u_type)
    elseif over_safe && algorithm.solve_options.under_safe && algorithm.solve_options.fp_sound
            algorithm.sum_fn = x -> safe_sum(x, algorithm.solve_options.u_type; under_safe=true)
    elseif !over_safe && !algorithm.solve_options.under_safe && algorithm.solve_options.fp_sound
            algorithm.sum_fn = x -> safe_sum_base(x, algorithm.solve_options.u_type)
    end

end


function model_network_layer!(layer::DotFunctor{ReLUType}, algorithm::Algorithm)
    res = similar(algorithm.prop_expr)
    f_unstable = (x::Interval{<:Real}) -> (sup(x) > 0 && inf(x) < 0)

    for i in eachindex(algorithm.prop_expr)
        res[i] = Flux.relu(algorithm.prop_expr[i])
        algorithm.results[:UnstableNodes] += f_unstable(algorithm.prop_expr[i])
    end

    algorithm.prop_expr = res
end


function model_network_layer!(layer::Flux.Dense, algorithm::Algorithm)
    res, risk = apply_dense(algorithm.prop_expr, layer, algorithm.solve_options.u_type, algorithm.solve_options.fp_sound; implementation=algorithm.solve_options.backend, sum_fn=algorithm.sum_fn)

    algorithm.prop_expr = res
    algorithm.results[:UnderOverRisk] = algorithm.results[:UnderOverRisk] || risk
end


function model_network_layer!(layer::Flux.Conv, algorithm::Algorithm)
    res, risk = apply_conv(algorithm.prop_expr, layer, algorithm.solve_options.u_type, algorithm.solve_options.fp_sound; implementation=algorithm.solve_options.backend, sum_fn=algorithm.sum_fn)

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
    model_network_layers!(algorithm::FPSoundIBP.Algorithm) -> Array{Interval{T}}

Propagate the interval bounds through every layer and return the output bounds:
"""
function FPSV.model_network_layers!(algorithm::Algorithm)
    for layer in algorithm.flattened_network
        model_network_layer!(layer, algorithm)
    end

    return algorithm.prop_expr
end

"""
    decide_label_uniformity!(algorithm::FPSoundIBP.Algorithm)

Compute the status from the output intervals `o` (stored in `results[:OutputBounds]`):
`PossibleUnderOrOverflow` if `results[:UnderOverRisk]` is set; otherwise `Adversarial` if
`sup(o_i - o_t) > 0` for some class `i ≠ t` (these `i` are stored in
`results[:HostileIndices]`), and `Safe` if there is no such class. `t` is `true_index`. See
[`Decision`](@ref).
"""
function FPSV.decide_label_uniformity!(algorithm::Algorithm)
    out_bounds = algorithm.prop_expr |> vec

    algorithm.results[:OutputBounds] = out_bounds

    out_bounds = out_bounds .- out_bounds[algorithm.true_index]
 
    algorithm.results[:HostileIndices] = []

    if algorithm.results[:UnderOverRisk]
        algorithm.results[:Status] = PossibleUnderOrOverflow
        return
    end

    for i in eachindex(out_bounds)
        if i != algorithm.true_index && sup(out_bounds[i]) > 0
            # It is just PossiblyAdversarial, since overestimation is significant in this algorithm.
            algorithm.results[:Status] = Adversarial
            push!(algorithm.results[:HostileIndices], i)
        end
    end

    if isempty(algorithm.results[:HostileIndices])
        algorithm.results[:Status] = Safe
    end

    return
end

end # module FPSoundIBP
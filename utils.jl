export Decision, Unknown, Safe, PossibleUnderOrOverflow, Adversarial
export normalize_network_layers, DotFunctor, f_interval_max
export FlattenType, LinearLayer, ReLUType, NormalizeLayerType
export get_input_domain
export assert_type!

import Setfield: @set

# Decision outputs
"""
    Decision

Result of a verification run, stored in `results[:Status]`.

- `Safe`: robustness is proven. For every output `i` other than the correct class `t`, the
  upper bound of `o_i - o_t` is at most `0`, where `o` are the output bounds over the whole
  input box.
- `Adversarial`: robustness could not be proven. This status is returned in two cases:
  (a) the unperturbed input is already misclassified (`PredictedIndex ≠ TrueIndex`); the
  verifier is then not run and no output bounds are stored; (b) for some other class `i`
  the upper bound of `o_i - o_t` is positive (these classes are listed in
  `results[:HostileIndices]`). Since the bounds over-approximate, (b) does not mean that an
  adversarial example exists.
- `PossibleUnderOrOverflow`: the overflow check (`over_safe = true`) found that some neuron
  may overflow for some input in the box. No robustness claim is made. Despite the name, only
  the overflow check sets this status; possible underflow is handled by widening the bounds
  (`under_safe = true`).
- `Unknown`: not produced by the current verifiers.

# What `Safe` guarantees

First, this library assumes the `textbook` implementation of the layers, other implemantions may require different extensions of our widening process.

| Options (flags of `run_tests.jl`) | `Safe` means robustness of … |
|:--- |:--- |
| `fp_sound = false` | the network evaluated in exact real arithmetic only; nothing follows for floating-point execution |
| `fp_sound = true` (`-s`) | the network evaluated in floating point with precision `u_type`, for every summation order and any rounding mode, provided that no overflow occurs and subnormal numbers are supported |
| `fp_sound`, `over_safe` (`-s -o`) | as above, and overflow is excluded |
| `fp_sound`, `over_safe`, `under_safe` (`-s -o -u`) | as above, also in flush-to-zero / denormals-are-zero environments |

In all cases the verified precision `u_type` must equal the deployment precision.
"""
@enum Decision begin
    Safe
    PossibleUnderOrOverflow
    Unknown
    Adversarial 
end


function f_interval_max(lhs::Interval, rhs::Interval)::Tuple{Interval, Bool}
    # transfer inputs to magical monad-land
    return f_interval_max((lhs, true), rhs)
end

function f_interval_max(monad::Tuple{Interval, Bool}, iv::Interval)::Tuple{Interval, Bool}
    if intersect(monad[1],iv) == ∅
        if monad[1] > iv
            monad
        else
            return (iv, true)
        end
    else
        return (union(monad[1], iv), false)
    end
end

function flatten_network(netlayer::Flux.Chain)
    sublayers = [flatten_network(sublayer) for sublayer in netlayer]
    flat_sublayers = reduce((arr,v) -> (typeof(v) <: Vector) ? push!(arr, v...) : push!(arr, v), sublayers, init=[])
    return flat_sublayers
end

# nullcase, an unknown layer is left alone
function flatten_network(netlayer)
    return netlayer
end


struct DotFunctor{T <: Function}
    f::T
end
(ftor::DotFunctor{T})(arg) where T = ftor.f.(arg)


function functorize_layer(layer)
    non_vectorized_functions = [typeof(Flux.flatten)]
    if typeof(layer) <: Function && !(typeof(layer) in non_vectorized_functions)
        # `x -> f(x)` type functions must be vectorized
        # (currently flatten is the only function that does not need vectorization)
        return DotFunctor(layer)
    else
        return layer
    end
end


function unwrap_embedded_activation!(unwrapped_layers::Vector{Any}, layer::Union{Flux.Dense, Flux.Conv})
    if typeof(layer.σ) == typeof(Flux.identity)
        push!(unwrapped_layers, layer)
    else
        activation = layer.σ
        layer = @set layer.σ = Flux.identity # creates a copy of the immutable struct, but with the given change
        push!(unwrapped_layers, layer)
        push!(unwrapped_layers, activation)
    end
end

function unwrap_embedded_activation!(unwrapped_layers::Vector{Any}, layer)
    push!(unwrapped_layers, layer)
end


"""
Flattens the network and applies function vectorization where needed.
"""
function normalize_network_layers(network::Flux.Chain)
    flat_layers = flatten_network(network)
    unwrapped_layers = []
    foreach(layer -> unwrap_embedded_activation!(unwrapped_layers, layer), flat_layers)
    functored_layers = map(functorize_layer, unwrapped_layers)
    return functored_layers
end

const LinearLayer = Union{Flux.Dense, Flux.Conv} # others can be added, but algorithms might not support them
const ReLUType = typeof(Flux.relu)
const FlattenType = typeof(Flux.flatten)
const NormalizeLayerType = typeof(NormalizeLayer)


function get_input_domain(input_point::AbstractArray{T}, perturbation_radius::T, input_range::Tuple{T,T}) where T<:Real
    # c_radius = convert(T, perturbation_radius)

    input_lower_domain = max.(input_point .- perturbation_radius, input_range[1])
    input_upper_domain = min.(input_point .+ perturbation_radius, input_range[2])
    
    return interval.(input_lower_domain, input_upper_domain)
end


function assert_type(x::Flux.Chain, ::Type{T}) where T <: Union{Float32, Float64}
    conversion = T == Float32 ? f32 : f64

    for l in x
        if (isa(l, Flux.Dense) || isa(l, Flux.Conv)) && (eltype(l.weight) != T || eltype(l.bias) != T)
            @warn "The network is not of type $T. Converting layer weights and biases to $T. You should consider defining the network with $T from the start!"
            return conversion(x)
        end
    end

    return x
end


function assert_type(x::AbstractArray{<:Real}, ::Type{T}) where T <: Union{Float32, Float64}

    if eltype(x) != T
        @warn "The input is not of type $T. Converting input to $T. You should consider defining the input with $T from the start!"
        return convert.(T, x)
    end

    return x
end


function assert_type(x::K, ::Type{T}) where {T <: Union{Float32, Float64}, K <: Real}
    if isa(x, T)
        return x
    else
        @warn "The perturbation radius is not of type $T. Converting perturbation radius to $T. You should consider defining the perturbation radius with $T from the start!"
        return convert(T, x)
    end
end


function assert_type!(alg)
    alg.network = assert_type(alg.network, alg.solve_options.u_type)
    alg.input_point = assert_type(alg.input_point, alg.solve_options.u_type)
    alg.perturbation_radius = assert_type(alg.perturbation_radius, alg.solve_options.u_type)

    return 
end
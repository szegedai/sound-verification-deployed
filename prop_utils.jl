export apply_dense, apply_conv

#############################
# Layer Transformations #####
#############################

############################
# Linear (Dense) Layer #####
############################
function textbook_dense(x::AbstractArray{T}, layer::Flux.Dense) where T <: Union{IntervalAffExpr, Interval}
    return textbook_linear(layer.weight, x, layer.bias)
end


function textbook_fp_sound_dense(x::AbstractArray{<:Interval}, layer::Flux.Dense, sum_fn::F) where {F}
    return textbook_linear(layer.weight, x, layer.bias; sum_fn=sum_fn)
end


function textbook_fp_sound_dense(x::AbstractArray{<:IntervalAffExpr}, layer::Flux.Dense, sum_fn::F) where {F}
    return textbook_linear(layer.weight, x, layer.bias; sum_fn=sum_fn)
end

##############
# Conv Layer #
##############
function textbook_conv(x::AbstractArray{T, 4}, layer::Flux.Conv) where T <: Union{IntervalAffExpr, Interval}
    return nnlib_conv(layer, x)
end


function textbook_fp_sound_conv(x::AbstractArray{<:Interval, 4}, layer::Flux.Conv, sum_fn::F) where {F}
    return nnlib_conv(layer, x; sum_fn=sum_fn)
end


function textbook_fp_sound_conv(x::AbstractArray{<:IntervalAffExpr, 4}, layer::Flux.Conv, sum_fn::F) where {F}
    return nnlib_conv(layer, x; sum_fn=sum_fn)
end


#############################
# Interface functions #######
#############################
"""
    apply_dense(x, layer::Flux.Dense, u_type, fp_sound; implementation = "textbook", sum_fn, vm) -> (y, overflow)

Propagate the bounds `x` (intervals or interval linear expressions) through the affine part
of `layer`. The activation of the layer is handled as a separate layer (see
[`normalize_network_layers`](@ref)).

- `fp_sound = true`: the neuron sums are computed with `sum_fn`, which the verifier sets to
  [`safe_sum`](@ref) or [`safe_sum_base`](@ref) for precision `u_type`.
- `fp_sound = false`: plain outward-rounded sums, i.e. bounds for the layer evaluated in real
  arithmetic. `sum_fn` is ignored and `overflow` is always `false`.

Only `implementation = "textbook"` is supported.
"""
function apply_dense(x::AbstractArray{T}, layer::Flux.Dense, u_type::Type, fp_sound::Bool; implementation::String="textbook", sum_fn::F = x -> (sum(x), false), vm::VariableManager = VariableManager())  where {F,T <: Union{IntervalAffExpr, Interval}}
    if implementation == "textbook"
        if fp_sound
            return textbook_fp_sound_dense(x, layer, sum_fn)
        else
            return textbook_dense(x, layer)
        end
    else
        throw(ArgumentError("Unsupported backend implementation: $implementation"))
    end

end


"""
    apply_conv(x, layer::Flux.Conv, u_type, fp_sound; implementation = "textbook", sum_fn, vm) -> (y, overflow)

Propagate the bounds `x` (intervals or interval linear expressions) through the affine part
of `layer`. The activation of the layer is handled as a separate layer (see
[`normalize_network_layers`](@ref)).

- `fp_sound = true`: the neuron sums are computed with `sum_fn`, which the verifier sets to
  [`safe_sum`](@ref) or [`safe_sum_base`](@ref) for precision `u_type`.
- `fp_sound = false`: plain outward-rounded sums, i.e. bounds for the layer evaluated in real
  arithmetic. `sum_fn` is ignored and `overflow` is always `false`.

Only `implementation = "textbook"` is supported.
"""
function apply_conv(x::AbstractArray{T}, layer::Flux.Conv, u_type::Type, fp_sound::Bool; implementation::String="textbook", sum_fn::F = x -> (sum(x), false), vm::VariableManager = VariableManager())  where {F,T <: Union{IntervalAffExpr, Interval}}
    if implementation == "textbook"
        if fp_sound
            return textbook_fp_sound_conv(x, layer, sum_fn)
        else
            return textbook_conv(x, layer)
        end
    else
        throw(ArgumentError("Unsupported backend implementation: $implementation"))
    end
end
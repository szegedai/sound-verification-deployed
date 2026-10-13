using IntervalArithmetic

export safe_sum, safe_sum_base

iszero_interval(x::Interval{T}) where T = inf(x) == sup(x) == T(0)
nonzero_interval(x::Interval{T}) where T = !iszero_interval(x)

concretize(x::IntervalAffExpr, vars::VariableManager) = x(vars)


"""
    compute_delta_interval(n, u_type, out_type) -> Interval{out_type}

Return the interval `[1 - Δ, 1 + Δ]` with `Δ = n μ / (1 - n μ)`, where `μ = eps(u_type)` is
the machine epsilon of the deployment precision (Lemma 1 and Eq. (4) of the paper).

The computation uses outward rounding in `out_type`, so the result encloses the exact
interval. For `n = 0` the result is `[1, 1]`. Throws an `ArgumentError` if `n μ ≥ 1`.
"""
function compute_delta_interval(n::Int, u_type::Type, out_type::Type)
    n = interval(out_type, n)
    u = interval(out_type, eps(u_type))

    nu = n*u
    if sup(nu) >= 1.0
        throw(ArgumentError("The number of summands exceed the maximum value (condition of n*μ < 1)"))
    end

    Delta = nu / (ExactReal(1) - nu)

    return interval(ExactReal(1) - Delta, ExactReal(1) + Delta)
end


"""
    under_widener(x::Interval{T}) -> Interval{T}

Underflow guard for one addend, used by [`safe_sum`](@ref) with `under_safe = true`
(Appendix B.2.2 of the paper, flush-to-zero / denormals-are-zero environments).

If each endpoint of `x` is either zero or at least `floatmin(T) / eps(T)` in absolute value,
`x` is returned unchanged; such endpoints are integer multiples of `floatmin(T)`, so sums of
them cannot underflow. Otherwise a message is printed and `x + [-floatmin(T), floatmin(T)]`
is returned.
"""
function under_widener(x::Interval{T})::Interval{T} where T <: AbstractFloat
    abs_bound = floatmin(T) / eps(T)

    if (abs(inf(x)) >= abs_bound || iszero(inf(x)) ) && (abs(sup(x)) >= abs_bound || iszero(sup(x)))
        return x
    else
        println("(Underflow correction) Dangerous interval $x. Widening is applied.")
        return x + interval(-floatmin(T), floatmin(T))
    end
end


"""
    bound_real_bounds(exp::IntervalAffExpr{T}, vm) -> (lower::Interval{T}, upper::Interval{T})

Bounds of the concretization of `exp` over the variable domains in `vm`, each returned as an
interval: `lower` encloses a real lower bound and `upper` a real upper bound of the values of
`exp`. These are the intervals checked against the underflow range in Appendix B.2.2.
Throws an `ArgumentError` if a variable of `exp` is not registered in `vm`.
"""
function bound_real_bounds(exp::IntervalAffExpr{T}, vm::VariableManager) where T <: AbstractFloat
    lower = interval(T, inf(exp.constant))
    upper = interval(T, sup(exp.constant))
    
    @inbounds for (key, coeff) in exp.coeffs
        if haskey(vm.vars, key)
            _res = coeff * vm.vars[key]
            lower += ExactReal(inf(_res))
            upper += ExactReal(sup(_res))
        else
            throw(ArgumentError("Variable $key not found in the provided VariableManager."))
        end
    end
    
    return (lower, upper)
end


"""
    concretize_and_widen(exp::IntervalAffExpr{T}, vm) -> (bounds::Interval{T}, exp_out::IntervalAffExpr{T})

Underflow guard for one symbolic addend, used by [`safe_sum`](@ref) with `under_safe = true`
(Appendix B.2.2 of the paper).
"""
function concretize_and_widen(exp::IntervalAffExpr{T}, vm::VariableManager)::Tuple{Interval{T}, IntervalAffExpr{T}} where T <: AbstractFloat
    l, u = bound_real_bounds(exp, vm)

    danger_domain = interval(-floatmin(T) / eps(T), floatmin(T) / eps(T))

    if (!isempty_interval(intersect_interval(l, danger_domain)) && nonzero_interval(l) ) || ( !isempty_interval(intersect_interval(u, danger_domain)) && nonzero_interval(u) )
        println("(Underflow correction) Dangerous interval [$l, $u]. Widening is applied.")

        return (interval(T, inf(l), sup(u)), IntervalAffExpr{T}(exp.constant + interval(-floatmin(T), floatmin(T)), exp.coeffs))
    else
        return (interval(T, inf(l), sup(u)), exp)
    end

end


# Sum functions
"""
    safe_sum(input::AbstractArray{Interval{T}}, u_type; under_safe = false) -> (s::Interval{T}, overflow::Bool)

Floating-point sound sum of the addends of one neuron (FPSoundIBP; Eq. (5) and
Proposition 2 of the paper).

`input` holds the addends `θ_1 x_1, …, θ_n x_n` of an inner product, each computed as an
outward-rounded interval product, and the bias. The returned interval `s` contains the result
of evaluating this inner product in floating-point precision `u_type`, including the rounding
of the products, for **every** expression tree (any order and parenthesisation of the
additions) and any rounding mode. It also contains the exact real value.

Each addend is multiplied by `[1 - Δ, 1 + Δ]` (see [`compute_delta_interval`](@ref)), where `n`
is the number of addends that are not exactly `[0, 0]`; an exact zero adds no rounding error.

`overflow` is `true` if the sum of the positive parts or the sum of the negative parts of the
widened addends reaches `Inf` or `-Inf` in `T` (Appendix B.1). If it is `false`, no
intermediate result of any expression tree can overflow.

With `under_safe = true`, each addend is first passed through [`under_widener`](@ref), so the
result stays sound in flush-to-zero / denormals-are-zero environments (Appendix B.2.2). With
subnormal numbers enabled (the IEEE 754 default) this is not needed.

Assumes that the verification precision `T` equals the deployment precision `u_type`; the
overflow test is only valid in that case.
"""
function safe_sum(input::AbstractArray{Interval{T}}, u_type::Type; under_safe::Bool=false)::Tuple{Interval{T}, Bool} where {T <: AbstractFloat}
    
    # Underflow correction
    ################################
    if under_safe
        input = under_widener.(input)
    end
    ################################

    l_pos = zero(Interval{T})
    l_neg = zero(Interval{T})
    u_pos = zero(Interval{T})
    u_neg = zero(Interval{T})

    poss_over = false

    # n_nz = sum(nonzero_interval.(input))
    n_nz = count(nonzero_interval, input)
    Delta = compute_delta_interval(n_nz, u_type, T)

    @inbounds for x in input
        l = inf(x)
        u = sup(x)

        if l > 0
            l_pos += interval(T, l) * Delta
        elseif l < 0
            l_neg += interval(T, l) * Delta
        end

        if u > 0
            u_pos += interval(T, u) * Delta
        elseif u < 0
            u_neg += interval(T, u) * Delta
        end
    end

    if isinf(sup(l_pos)) || isinf(sup(u_pos)) || isinf(inf(l_neg)) || isinf(inf(u_neg))
        poss_over = true
    end

    final_inf = inf(l_pos + l_neg)
    
    final_sup = sup(u_pos + u_neg)

    return (interval(final_inf, final_sup), poss_over)
end


"""
    safe_sum(input::AbstractArray{IntervalAffExpr{T}}, u_type, vm; under_safe = false) -> (s::IntervalAffExpr{T}, overflow::Bool)

Symbolic counterpart of the interval method (FPSoundSymbolic; Eq. (7) and Proposition 3 of
the paper).

Each addend expression `θ_j L_j(x)` is multiplied coefficient-wise by `[1 - Δ, 1 + Δ]`, and
the expressions are added. `n` is the number of addends whose concretization over `vm` is not
exactly `[0, 0]`. The overflow flag is computed from the concretized addends in the same way
as in the interval method.

With `under_safe = true`, each addend is checked with [`concretize_and_widen`](@ref), which
widens its constant term if its bounds may fall into the underflow range (Appendix B.2.2).

!!! warning
    The expressions in `input` are scaled in place (see [`inplace_scale!`](@ref)) and must not
    be used afterwards.
"""
function safe_sum(input::AbstractArray{IntervalAffExpr{T}}, u_type::Type, vm::VariableManager; under_safe::Bool=false) where T <: AbstractFloat
    
    # Underflow correction
    ################################################################
    if under_safe
        _res = concretize_and_widen.(input, Ref(vm))

        concretized_input = first.(_res)
        input = last.(_res)
    else
        concretized_input = concretize.(input, Ref(vm))
    end
    ################################################################

    l_pos = zero(Interval{T})
    l_neg = zero(Interval{T})
    u_pos = zero(Interval{T})
    u_neg = zero(Interval{T})

    poss_over = false

    n_nz = count(nonzero_interval, concretized_input)
    Delta = compute_delta_interval(n_nz, u_type, T)

    input .= inplace_scale!.(input, Ref(Delta))
    
    final_exp = sum(input)


    @inbounds for x in concretized_input
        l = inf(x)
        u = sup(x)

        if l > 0
            l_pos += interval(T, l) * Delta
        elseif l < 0
            l_neg += interval(T, l) * Delta
        end

        if u > 0
            u_pos += interval(T, u) * Delta
        elseif u < 0
            u_neg += interval(T, u) * Delta
        end
    end

    if isinf(sup(l_pos)) || isinf(sup(u_pos)) || isinf(inf(l_neg)) || isinf(inf(u_neg))
        poss_over = true
    end
    
    return (final_exp, poss_over)
end


"""
    safe_sum_base(input::AbstractArray{Interval{T}}, u_type) -> (s::Interval{T}, false)
    safe_sum_base(input::AbstractArray{IntervalAffExpr{T}}, u_type, vm) -> (s::IntervalAffExpr{T}, false)

Like [`safe_sum`](@ref), but without overflow and underflow handling: returns the sum of the
addends multiplied by `[1 - Δ, 1 + Δ]` and always reports `false` for overflow. Used when
`fp_sound = true` and `over_safe = false`.
"""
function safe_sum_base(input::AbstractArray{Interval{T}}, u_type::Type)::Tuple{Interval{T}, Bool} where {T <: AbstractFloat}
    n_nz = count(nonzero_interval, input)
    Delta = compute_delta_interval(n_nz, u_type, T)

    return (sum(input .* Delta), false)
end


function safe_sum_base(input::AbstractArray{IntervalAffExpr{T}}, u_type::Type, vm::VariableManager; under_safe::Bool=false) where T <: AbstractFloat
    concretized_input = concretize.(input, Ref(vm))

    n_nz = count(nonzero_interval, concretized_input)
    Delta = compute_delta_interval(n_nz, u_type, T)

    input .= inplace_scale!.(input, Ref(Delta))
    
    return (sum(input), false)
end
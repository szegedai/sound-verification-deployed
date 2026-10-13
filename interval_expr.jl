export IntervalExpr, IntervalAffExpr
export extend_exp_with_new_var!
export VariableManager, register_new_var!, get_next_var_sym
export inplace_scale!

########################################
########### Variable Manager ###########
########################################

# Struct for managing variables
"""
    VariableManager{S, T}

Registry of the variables of [`IntervalExpr`](@ref)s: maps every variable name (of type `S`,
normally `Symbol`) to its domain, an `Interval{T}`.

Variables are named `:x_0, :x_1, …` in the order of registration. In `FPSoundSymbolic` the
first variables are the input coordinates, and every unstable ReLU neuron adds one more
variable (see `FPSoundSymbolic.apply_relu_on_exp`).

    VariableManager(first_var_id, T)   # empty registry; the first variable is named x_<first_var_id>
    VariableManager()                  # VariableManager(0, Float64)
"""
mutable struct VariableManager{S, T<:Real}
    vars::OrderedDict{S, Interval{T}}
    next_var_ID::Int64
end


VariableManager(first_var_id::Int, ::Type{T}) where T = VariableManager{Symbol, T}(OrderedDict{Symbol, T}(), first_var_id)
VariableManager() = VariableManager(0, Float64)


"""
    register_new_var!(vm, domain) -> Symbol

Register a new variable with domain `domain` in `vm` and return its name.
"""
function register_new_var!(vm::VariableManager{S, T}, domain::Interval{T}) where {S, T<:Real}
    new_var = Symbol("x_", vm.next_var_ID)
    vm.vars[new_var] = domain
    vm.next_var_ID += 1
    return new_var
end


"""
    get_next_var_sym(vm) -> Symbol

Name that the next variable registered in `vm` will get. Does not register anything.
"""
get_next_var_sym(vm::VariableManager{S, T}) where {S, T<:Real} = Symbol("x_", vm.next_var_ID)


########################################
######### Interval Expression ##########
########################################

Flux.relu(iv::Interval{T}) where T = interval(iv |> inf |> Flux.relu, iv |> sup |> Flux.relu)

# IntervalExpr struct for interval linear expressions
"""
    IntervalExpr{S, T}
    IntervalAffExpr{T} = IntervalExpr{Symbol, T}

Interval linear expression

    L(x) = C + c_1 x_1 + … + c_k x_k

with interval constant `C = constant` and interval coefficients `c_i = coeffs[x_i]` (Eq. (2)
of the paper).

Calling an expression on the variable domains concretizes it with outward-rounded interval
arithmetic, `C + c_1 dom(x_1) + … + c_k dom(x_k)`:

    L(vm::VariableManager)
    L(domains::OrderedDict{S, Interval{T}})

Supported operations: `+` with another expression or with a constant, `*` with a constant
(an interval or a number; the product of two non-constant expressions is not supported),
`sum` and [`inplace_scale!`](@ref). They act coefficient-wise with outward rounding.

Note: `L + c` for a constant `c` shares the coefficient dictionary with `L`.
"""
struct IntervalExpr{S, T<:Real}
    constant::Interval{T}
    coeffs::OrderedDict{S, Interval{T}}
end

# The main type for interval linear expressions
IntervalAffExpr{T} = IntervalExpr{Symbol, T} where T<:Real

IntervalExpr(constant::Interval{T}) where T <: Real = IntervalExpr{Symbol, T}(convert(Interval{T}, constant), OrderedDict{Symbol, Interval{T}}())
IntervalExpr(constant::T) where T <: Real = IntervalExpr(interval(T, constant))

IntervalExpr{Symbol, T}(constant::Interval{<: Real}) where T <: Real = IntervalAffExpr{T}(constant, OrderedDict{Symbol, Interval{T}}())
IntervalExpr{Symbol, T}(constant::Real) where T <: Real = IntervalAffExpr{T}(interval(T, constant), OrderedDict{Symbol, Interval{T}}())


# Concretization function for interval linear expressions
function (exp::IntervalExpr{S, T})(x::OrderedDict{S, Interval{T}}) where {S, T<:Real}
    result = exp.constant
    for (key, coeff) in exp.coeffs
        if haskey(x, key)
            result += coeff * x[key]
        else
            throw(ArgumentError("Variable $key not found in the provided dictionary."))
        end
    end
    return result
end

(exp::IntervalExpr{S, T})(x::VariableManager{S, T}) where {S, T<:Real} = exp(x.vars)

# Operators for interval linear expressions
# Addition
function Base.:+(a::IntervalExpr{S, T}, b::IntervalExpr{S, T}) where {S, T<:Real}
    new_constant = a.constant + b.constant
    new_coeffs = OrderedDict(a.coeffs)
    for (key, coeff) in b.coeffs
        if haskey(new_coeffs, key)
            new_coeffs[key] += coeff
        else
            new_coeffs[key] = coeff
        end
    end
    return IntervalExpr{S, T}(new_constant, new_coeffs)
end

Base.:+(a::IntervalExpr{S, T}, b::Union{Interval{T}, T}) where {S, T<:Real} = IntervalExpr{S, T}(a.constant + interval(T, b), a.coeffs)
Base.:+(a::Union{Interval{T}, T}, b::IntervalExpr{S, T}) where {S, T<:Real} = b + a

# Multiplication
function Base.:*(a::IntervalExpr{S, T}, b::Union{Interval{T}, T, ExactReal{T}}) where {S, T<:Real}
    b = interval(T, b)
    new_constant = a.constant * b
    new_coeffs = OrderedDict{S, Interval{T}}()

    if !iszero(b)
        for (key, coeff) in a.coeffs
            new_coeffs[key] = coeff * b
        end
    end

    return IntervalExpr{S, T}(new_constant, new_coeffs)
end

Base.:*(a::Union{Interval{T}, T, ExactReal{T}}, b::IntervalExpr{S, T}) where {S, T<:Real} = b * a

function Base.:*(a::IntervalExpr{S,T}, b::IntervalExpr{S, T}) where {S, T <: Real}
    if !(is_constant(a) || is_constant(b))
        throw(ArgumentError("Multiplication of two IntervalExprs is not supported unless one is constant."))
    end

    if is_constant(a)
        return b * a.constant
    else
        return a * b.constant
    end

end

Base.:*(a::Bool, b::IntervalExpr{S, T}) where {S, T<:Real} = b * (a ? 1.0 : 0.0)

is_constant(exp::IntervalExpr{S, T}) where {S, T<:Real} = isempty(exp.coeffs)

# Additional methods for zero and one
Base.zero(::Type{IntervalExpr{S, T}}) where {S, T<:Real} = IntervalExpr{S, T}(interval(T, 0.0))
Base.zero(x::IntervalExpr{S, T}) where {S, T<:Real} = zero(typeof(x))

Base.one(::Type{IntervalExpr{S, T}}) where {S, T<:Real} = IntervalExpr{S, T}(interval(T, 1.0))
Base.one(x::IntervalExpr{S, T}) where {S, T<:Real} = one(typeof(x))

# Conversion and promotion rules
Base.convert(::Type{IntervalExpr{S, T}}, x::Interval{<:Real}) where {S, T<:Real} = IntervalExpr{S, T}(x)
Base.convert(::Type{IntervalExpr{S, T}}, x::Real) where {S, T<:Real} = IntervalExpr{S, T}(interval(T, x))

Base.promote_rule(::Type{IntervalExpr{S, T}}, ::Type{Interval{T}}) where {S, T<:Real} = IntervalExpr{S, T}
Base.promote_rule(::Type{IntervalExpr{S, T}}, ::Type{T}) where {S, T<:Real} = IntervalExpr{S, T}

Base.iszero(x::IntervalExpr{S, T}) where {S, T<:Real} = iszero(x.constant) && isempty(x.coeffs)

# Show method for printing
function Base.show(io::IO, exp::IntervalExpr{S, T}) where {S, T<:Real}
    print(io, "IntervalExpr(constant: ", exp.constant, ", coeffs: {")
    first = true
    for (key, coeff) in exp.coeffs
        if !first
            print(io, ", ")
        end
        print(io, "$key => $coeff")
        first = false
    end
    print(io, "})")
end


"""
    extend_exp_with_new_var!(exp, new_var, coeff) -> exp

Add the term `coeff * new_var` to `exp` in place. Throws an `ArgumentError` if `exp` already
contains `new_var`.
"""
function extend_exp_with_new_var!(exp::IntervalExpr{S, T}, new_var::S, coeff::Interval{T}) where {S, T<:Real}
    if haskey(exp.coeffs, new_var)
        throw(ArgumentError("Variable $new_var already exists in the expression."))
    else
        exp.coeffs[new_var] = coeff
    end
    return exp
end


function Base.sum(arr::AbstractArray{IntervalExpr{S, T}}) where {S, T<:Real}
    if isempty(arr)
        return zero(IntervalExpr{S, T})
    end

    total_coeffs = OrderedDict{S, Interval{T}}()
    total_const = zero(Interval{T})

    @inbounds for exp in arr
        total_const += exp.constant
        
        for (key, coeff) in exp.coeffs
            if haskey(total_coeffs, key)
                total_coeffs[key] += coeff
            else
                total_coeffs[key] = coeff
            end
        end
    end

    return IntervalExpr{S, T}(total_const, total_coeffs)
end


"""
    inplace_scale!(exp, factor) -> IntervalExpr

Return `factor * exp`, reusing the coefficient dictionary of `exp`: every coefficient is
multiplied by `factor` in place, and a new expression with the scaled constant is returned.
`exp` itself is left with scaled coefficients but the old constant, so only the returned value
may be used afterwards. If `factor` is zero, all terms are removed.

Used by [`safe_sum`](@ref) to multiply each addend by `[1 - Δ, 1 + Δ]` (Eq. (7) of the paper).
"""
function inplace_scale!(exp::IntervalExpr{S, T}, factor::Union{Interval{T}, T}) where {S, T<:Real}
    factor = interval(T, factor)
    
    if iszero(factor)
        empty!(exp.coeffs)
        return IntervalExpr{S, T}(exp.constant * factor, exp.coeffs)
    end

    for (key, coeff) in exp.coeffs
        exp.coeffs[key] = coeff * factor
    end

    return IntervalExpr{S, T}(exp.constant * factor, exp.coeffs)
end
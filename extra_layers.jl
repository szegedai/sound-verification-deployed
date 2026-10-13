export NormalizeLayer

"""
    NormalizeLayer(mean, std)
"""
struct NormalizeLayer{T}
    mean::Array{T, 4}
    std::Array{T, 4}
end

(l::NormalizeLayer)(x) = (x .- l.mean) ./ l.std

function (l::NormalizeLayer)(x::AbstractArray{Interval{T}}) where T
    iv_mean = interval.(T, l.mean)
    iv_std = interval.(T, l.std)

    return (x .- iv_mean) ./ iv_std
end


function (l::NormalizeLayer)(x::AbstractArray{IntervalAffExpr{T}}) where T 
    iv_mean = -interval.(T, l.mean)
    iv_std = interval(T, 1) ./ interval.(T, l.std)

    return (x .+ iv_mean) .* iv_std
end



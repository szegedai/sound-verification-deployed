###################
# Linear layer ####
###################
"""
    textbook_linear(A, B, bias; sum_fn = x -> (sum(x), false)) -> (C, overflow)

Bound propagation through the affine map `C = A * B .+ bias`, where `A` is a floating-point
weight matrix and `B` holds intervals or interval linear expressions. Vectors are treated
as one-column matrices, so `C` is always a matrix.

For each output entry `C[i, j]` the addends `A[i, 1] * B[1, j], …, A[i, k] * B[k, j]` and
`bias[i]` are collected in a buffer and reduced with `sum_fn`, which must return a tuple
`(sum, overflow_flag)`. The weights are wrapped in `ExactReal`, i.e. they are treated as the
exact floating-point numbers used in deployment, and the products are computed with outward
rounding.

`sum_fn` determines the meaning of the result: the default `sum` gives bounds for the layer
evaluated in real arithmetic, [`safe_sum`](@ref) and [`safe_sum_base`](@ref) give
floating-point sound bounds.

Returns `C` and `true` if `sum_fn` reported a possible overflow for any entry.
"""
function textbook_linear(A::AbstractArray{T1}, B::AbstractArray{T2}, bias::AbstractArray{T3}; sum_fn::F = x -> (sum(x), false)) where {T1,T2,T3,F}
    A = ndims(A) == 1 ? A[:,:] : A
    B = ndims(B) == 1 ? B[:,:] : B
    
    return textbook_linear(A, B, bias; sum_fn=sum_fn)
end


function textbook_linear(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}, bias::AbstractVector{T3}; sum_fn::F = x -> (sum(x), false)) where {T1,T2,T3,F}
    m, k1 = size(A)
    k2, n = size(B)

    # Sanity checks
    @assert k1 == k2 "Matrix dimensions must agree"
    l_bias = length(bias)
    @assert m == l_bias "Expected bias length $n, got $l_bias"

    T = Base.promote_op(+, Base.promote_op(*, T1, T2), Base.promote_op(*, T1, T2))

    buffer = Vector{T}(undef, k1 + 1)

    C = Matrix{T}(undef, m, n)

    any_risk = false

    @inbounds for i in 1:m
        for j in 1:n
            buffer[end] = interval(bias[i])
            for k in 1:k1
                buffer[k] = ExactReal(A[i,k]) * B[k,j]
            end
            C[i,j], _risk = sum_fn(buffer)
            any_risk = any_risk || _risk
        end
    end

    return (C, any_risk)
end


###################
# Conv layer ######
###################
# `calc_padding_regions` and `nnlib_conv` are adapted from NNlib.jl
# (src/impl/padding_edges.jl and src/impl/conv_direct.jl, https://github.com/FluxML/NNlib.jl).
# NNlib.jl is licensed under the MIT "Expat" License:
#
#   Copyright (c) 2017-19: Julia Computing, Inc., Mike J Innes, and Contributors
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

"""
    calc_padding_regions(dims) -> (padded_regions, central_region)

Split the output positions `(w, h)` of a 2D convolution into the central region, whose
receptive fields lie entirely inside the input, and the border regions, whose receptive
fields overlap the zero padding. Taken from NNlib.jl.
"""
function calc_padding_regions(dims)
    width, height = dims.input_size
    kernel_w, kernel_h = dims.kernel_size
    C_in = dims.channels_in
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi = dims.padding
    dil_w, dil_h = dims.dilation
    stride_w, stride_h = dims.stride
    out_width, out_height = Flux.output_size(dims)

    # Let us first calculate the number of rows/cols within which we must zero out some
    # portion of the image patches we're copying over.  The "spillage" here is the number
    # of indices along a particular dimension for which a kernel will have some portion
    # of its input domain overlapping the padding.  If padding is zero, these values are
    # all trivially zero.  The low spillage is trivially the low padding divided by the
    # stride; literally the number of shifts that overlap some padding.  The high
    # spillage is slightly more complicated; we first figure out how many elements of
    # high padding are wasted (e.g. through strides not fitting to the end perfectly)
    # subtract that from the high padding, then do the same:
    calc_lo_spill(O, S, P) = max(min(ceil(Int, P/S), O),0)
    @inline function calc_hi_spill(O, S, Pl, Ph, K, D, I)
        wasted_Ph = (I + Pl + Ph - (K - 1)*D - 1)%S
        return max(min(ceil(Int, (Ph - wasted_Ph)/S), O), 0)
    end

    spill_w_lo = calc_lo_spill(out_width, stride_w, pad_w_lo)
    spill_w_hi = calc_hi_spill(out_width, stride_w, pad_w_lo, pad_w_hi, kernel_w, dil_w, width)
    spill_h_lo = calc_lo_spill(out_height, stride_h, pad_h_lo)
    spill_h_hi = calc_hi_spill(out_height, stride_h, pad_h_lo, pad_h_hi, kernel_h, dil_h, height)

    spill_w_hi_abs = out_width  - spill_w_hi + 1
    spill_h_hi_abs = out_height - spill_h_hi + 1

    # These are the regions we're going to have to run with cognizance of padding.
    # There are six of them; one for each face of the cube image.  We explicitly
    # design this so that we run over `width` most tightly, in the expectation that
    # this will generate better code for when `h` and `d` are singleton dimensions.
    # We visualize this as a cube, indexed by dimensions (w, h).
    padded_regions = (
        # The next largest chunk we choose will be the lower-h WD faces; we always
        # want to maximize going across full `w`, as its contiguous in memory.
        (
            1:out_width,
            1:spill_h_lo,
        ),
        # Then the upper-h WD face
        (
            1:out_width,
            spill_h_hi_abs:out_height,
        ),

        # Next, we fit the HD faces in, but without overlapping the `h` and `d`
        # regions we've done before:
        (
            1:spill_w_lo,
            (spill_h_lo+1):(spill_h_hi_abs-1),
        ),
        (
            spill_w_hi_abs:out_width,
            (spill_h_lo+1):(spill_h_hi_abs-1),
        ),
    )

    # The central region that has no padding.
    central_region = (
        (spill_w_lo+1):(spill_w_hi_abs - 1),
        (spill_h_lo+1):(spill_h_hi_abs - 1),
    )
    return padded_regions, central_region
end


"""
    nnlib_conv(conv::Flux.Conv, input::Array{T, 4}; sum_fn = x -> (sum(x), false), s_kernel = nothing, s_bias = nothing) -> (output, overflow)

Bound propagation through a 2D convolution layer without its activation. It computes the
same function as `conv(input)` in Flux (convolution with flipped kernel, `stride`, `pad`,
`dilation`); only `groups == 1` is supported.

`input` is a `W × H × C_in × batch` array of intervals or interval linear expressions. For
each output entry the addends `bias, x_1 w_1, …, x_k w_k` are collected and reduced with
`sum_fn`, exactly as in [`textbook_linear`](@ref). Input positions in the zero padding are left
out; they are exact zeros and add no rounding error. The weights are wrapped in `ExactReal`.

Returns the output array and `true` if `sum_fn` reported a possible overflow.

`s_kernel` and `s_bias` replace the weights and the bias of `conv` (for testing; give both or
neither). Adapted from `conv_direct!` of NNlib.jl (MIT license, see the notice above).
"""
function nnlib_conv(conv::Flux.Conv, input::Array{T, 4}; s_kernel=nothing, s_bias=nothing, sum_fn::F = x -> (sum(x), false)) where {T, F}
    if isnothing(s_kernel) && isnothing(s_bias)
       kernel = conv.weight
       bias = conv.bias
    else
        kernel = s_kernel
        bias = s_bias
    end

    # kernel = conv.weight
    stride_w, stride_h = conv.stride
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi = conv.pad
    dil_w, dil_h = conv.dilation
    @assert conv.groups == 1 "Limited implementation"

    input_size = size(input)
    (in_width, in_height, in_channels, batch) = input_size
    (kernel_width, kernel_height, kernel_in_channels, _) = size(kernel)

    @assert(
        in_channels == kernel_in_channels,
        "Number of channels in input, $in_channels, does not match number of channels, $kernel_in_channels, that kernels operate on."
    )

    output_size = Flux.outputsize(conv, input_size)
    (out_width, out_height, out_channels, batch) = output_size

    # W = typeof(kernel*input + bias) type calculations
    W = Base.promote_op(+, eltype(bias), Base.promote_op(*, eltype(kernel), eltype(input)))
    output = Array{W}(undef, output_size)

    # A helper function to project from output (w, h) to input (input_w, input_h)
    project(idx, stride, pad) = (idx - 1)*stride - pad + 1
    # Create a method that determines how we're going to index into the kernel.
    kproj(k, M) = M - k + 1
    
    # Use `calc_padding_regions` to determine where we do or don't need to worry about padding
    # cdims = DenseConvDims(size(input), size(kernel); stride, padding, conv.dilation, flipkernel=false, conv.groups)
    cdims = Flux.conv_dims(conv, input)
    # @info cdims
    padded_regions, central_region = calc_padding_regions(cdims)
    
    # Additional safety variable and buffer
    any_risk = false
    max_k_size = kernel_width * kernel_height * in_channels + 1
    buffer = Vector{W}(undef, max_k_size)
    # -------------------------------------


    # Start with the central region
    w_region, h_region = central_region
    @inbounds for b_idx in 1:batch,
        c_out in 1:out_channels,
        h_idx in h_region,
        w_idx in w_region

        # dotprod = muladd(bias[c_out], 1, zero(T))
        buffer[1] = interval(bias[c_out])
        buf_idx = 2

        for c_in in 1:in_channels,
            kh in 1:kernel_height,
            kw in 1:kernel_width

            # Hoist me, you coward.
            x_h = project(h_idx, stride_h, pad_h_lo) + (kh - 1)*dil_h
            x_w = project(w_idx, stride_w, pad_w_lo) + (kw - 1)*dil_w

            # Since we're in the central region, we don't need to worry about clamping
            x_val = input[x_w, x_h, c_in, b_idx]
            w_val = kernel[
                        kproj(kw, kernel_width),
                        kproj(kh, kernel_height),
                        c_in,
                        c_out
                    ]
            # dotprod = muladd(x_val, w_val, dotprod)
            buffer[buf_idx] = x_val * ExactReal(w_val)
            buf_idx += 1
        end

        val, _risk = sum_fn(buffer)
        output[w_idx, h_idx, c_out, b_idx] = val
        any_risk = any_risk || _risk
    end
    
    # Next, do potentially-padded regions:
    @inbounds for (w_region, h_region) in padded_regions
        for b_idx in 1:batch,
            c_out in 1:out_channels,
            h_idx in h_region,
            w_idx in w_region

            # dotprod = muladd(bias[c_out], 1, zero(T))
            buffer[1] = interval(bias[c_out])
            buf_idx = 2

            for c_in in 1:in_channels,
                kh in 1:kernel_height,
                kw in 1:kernel_width

                x_h = project(h_idx, stride_h, pad_h_lo) + (kh - 1)*dil_h
                x_w = project(w_idx, stride_w, pad_w_lo) + (kw - 1)*dil_w
                input_index = (x_w, x_h, c_in, b_idx)

                if checkbounds(Bool, input, input_index...)
                    x_val = input[input_index...]
                    w_val = kernel[
                                kproj(kw, kernel_width),
                                kproj(kh, kernel_height),
                                c_in,
                                c_out
                            ]
                    # dotprod = muladd(x_val, w_val, dotprod)
                    buffer[buf_idx] = x_val * ExactReal(w_val)
                    buf_idx += 1
                end
            end
            val, _risk = sum_fn(view(buffer, 1:buf_idx-1))
            output[w_idx, h_idx, c_out, b_idx] = val
            any_risk = any_risk || _risk
            #output[w_idx, h_idx, c_out, b_idx] = dotprod
        end
    end

    return output, any_risk
end

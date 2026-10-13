"""
    FPSoundVerification

Floating-point sound local robustness verification of ReLU networks.

A verification result is *floating-point sound* if it holds for the network as it is
executed in floating point: for every order (expression tree) in which a deployment
environment may evaluate the sums inside a neuron and for any rounding mode, and not only
for the network evaluated in exact real arithmetic.

Two verifiers with the same interface are provided as submodules:

- [`FPSoundIBP`](@ref): interval bound propagation (FPSoundIBP in the paper).
- [`FPSoundSymbolic`](@ref): symbolic propagation with interval linear expressions
  (FPSoundSymbolic in the paper).

Supported layers: `Flux.Dense`, `Flux.Conv` (`groups == 1`), `relu`, `Flux.flatten` and
[`NormalizeLayer`](@ref). The floating-point guarantees cover `Dense`, `Conv`, `relu` and
`flatten`; see [`NormalizeLayer`](@ref) for the normalisation layer.

All guarantees assume that the verified precision `u_type` equals the precision used in
deployment and that the weights of the network are exactly the floating-point numbers used
in deployment.
"""
module FPSoundVerification

macro iface_function(functions...)
    for f in functions
        eval(quote
            function $f end
            export $f
        end)
    end
end

# interface functions
@iface_function init_bounding_algorithm! model_network_layers! decide_label_uniformity!

"""
    init_bounding_algorithm!(algorithm)

Prepare `algorithm` for propagation:

1. convert the network, the input point and the radius to the verified precision
   `u_type` (see `assert_type!`);
2. build the input box (see `get_input_domain`);
3. split the network into elementary layers (see [`normalize_network_layers`](@ref));
4. select the neuron summation routine `algorithm.sum_fn` from the options `fp_sound`,
   `over_safe` and `under_safe`.

Implemented for `FPSoundIBP.Algorithm` and `FPSoundSymbolic.Algorithm`.
"""
init_bounding_algorithm!

"""
    model_network_layers!(algorithm)

Propagate the input box through all layers of the network and return the bounds of the
output neurons (intervals for `FPSoundIBP`, interval linear expressions for
`FPSoundSymbolic`). The bounds are also kept in `algorithm.prop_expr`.

During the propagation `algorithm.results[:UnderOverRisk]` is set to `true` if the overflow check reports a
possible overflow in any neuron.

Must be called after [`init_bounding_algorithm!`](@ref).
"""
model_network_layers!

"""
    decide_label_uniformity!(algorithm)

Decide whether the correct class has the largest output on the whole input box, using the
output bounds computed by [`model_network_layers!`](@ref). Sets the entries
`:OutputBounds`, `:HostileIndices` and `:Status` of `algorithm.results`; see
[`Decision`](@ref) for the possible statuses and their meaning.
"""
decide_label_uniformity!


using IntervalArithmetic
IntervalArithmetic.setdisplay(:infsup; ng_flag=false)

using Flux
using OrderedCollections: OrderedDict
using Configurations

export @option, from_dict, from_kwargs
export OrderedDict

include("interval_expr.jl")
include("extra_layers.jl")
include("utils.jl")
include("textbook_imp.jl")
include("sound_sum.jl")
include("prop_utils.jl")

abstract type AbstractOptions end
export AbstractOptions

include("ibp.jl")
export FPSoundIBP

include("symbolic.jl")
export FPSoundSymbolic

include("running_helper.jl")

end # module FPSoundVerification


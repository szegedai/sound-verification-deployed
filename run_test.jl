import Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "FPSoundVerification.jl"))
using .FPSoundVerification
using Flux
using ArgParse
using Random


# Parse the command line arguments
argparser = ArgParseSettings()

@add_arg_table argparser begin
    "--verifier", "-v"
        help = "Algorithm to use: IBP or Symbolic"
        arg_type = String
        default = "IBP"
    "--num_instance", "-n"
        help = "How many (random) instances to run the verification on"
        arg_type = Int
        default = 10
    "--perturbation_radius", "-p"
        help = "Perturbation radius"
        arg_type = Float64
        default = 0.01
    "--fp_sound", "-s"
        help = "Whether to use floating-point sound verification"
        action = :store_true
    "--over_safe", "-o"
        help = "Check for overflows"
        action = :store_true
    "--under_safe", "-u"
        help = "Check for underflows"
        action = :store_true
    "--backend", "-b"
        help = "Backend linear algebra implementation to use (only textbook is supported)"
        arg_type = String
        default = "textbook"
    "--verified_precision", "-r"
        help = "The verified precision: Float32 or Float64"
        arg_type = String
        default = "Float64"
    "--output_path", "-d"
        help = "Directory where the result CSV files are written"
        arg_type = String
        default = "./Results"
end

parsed_args = parse_args(argparser; as_symbols = true)
net_type = parsed_args[:verified_precision] == "Float32" ? Float32 : Float64


# A small predefined network (fixed random weights): 8×8×1 input, 3 classes.
# The weights must have the verified precision, hence the conversion to net_type.
Random.seed!(2026)
nn = Chain(
    Conv((3, 3), 1 => 4, relu; stride = 2),   # 8×8×1 -> 3×3×4
    Flux.flatten,
    Dense(36 => 16, relu),
    Dense(16 => 3),
)
nn = net_type == Float32 ? f32(nn) : f64(nn)


# A random dataset in the W × H × C × N layout, with values in the input range [0, 1].
# The correct classes are the predictions of the network, so that every input is verified.
num_instance = parsed_args[:num_instance]
dataset = rand(net_type, 8, 8, 1, num_instance)
output_indices = [argmax(vec(nn(dataset[:, :, :, k:k]))) for k in 1:num_instance]


# Determine the verifier and its corresponding algorithm to use
if parsed_args[:verifier] == "IBP"
    verifier = FPSoundIBP
elseif parsed_args[:verifier] == "Symbolic"
    verifier = FPSoundSymbolic
else
    error("Unknown algorithm: $(parsed_args[:verifier])")
end

alg = verifier.Algorithm

solve_options = Dict{String, Any}(
    "input_range" => net_type.((0.0, 1.0)),
    "u_type" => net_type,
    "fp_sound" => parsed_args[:fp_sound],
    "backend" => parsed_args[:backend],
    "over_safe" => parsed_args[:over_safe],
    "under_safe" => parsed_args[:under_safe]
);


# Run the verification on the dataset; one CSV row per input
run_on_batch(
    alg,
    nn,
    dataset,
    output_indices,
    net_type(parsed_args[:perturbation_radius]);
    # Keyword arguments
    network_id = "toy_net",
    solve_options = from_dict(verifier.Option, solve_options),
    output_path = parsed_args[:output_path]
)
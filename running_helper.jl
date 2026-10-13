using CPUTime
using MAT
using Dates
using Statistics

export run_on_sample, run_on_batch

"""
    run_on_sample(network, input_point, true_index, perturbation_radius, main_algorithm, solve_options) -> Dict{Symbol, Any}

Verify the local robustness of `network` around one input.

# Arguments
- `network::Flux.Chain`: the classifier; its output is the vector of class scores.
- `input_point`: the unperturbed input, a `W × H × C × 1` array.
- `true_index`: 1-based index of the correct class.
- `perturbation_radius`: radius of the L∞ ball around `input_point`.
- `main_algorithm`: `FPSoundIBP.Algorithm` or `FPSoundSymbolic.Algorithm` (without type
  parameter).
- `solve_options`: the matching `FPSoundIBP.Option` or `FPSoundSymbolic.Option`.

If `network` misclassifies `input_point`, the verifier is not run and the status is
`Adversarial`.

# Returned entries
- `:Status`: a [`Decision`](@ref).
- `:PredictedIndex`: `argmax` of `network(input_point)`, evaluated by Flux on the input as given.
- `:TrueIndex`, `:Radius`, `:VerifiedPrecision` (`u_type`), `:MainAlgorithm`,
  `:AlgorithmOptions`: the inputs of the run.
- `:SolveTime`: wall-clock time of the verification in seconds. The first call in a Julia
  session also includes JIT compilation.
- Only if the verifier was run: `:OutputBounds` (one interval per output),
  `:HostileIndices`, `:UnderOverRisk` (possible overflow detected) and `:UnstableNodes`
  (number of ReLU neurons whose input bounds contain both negative and positive values).
"""
function run_on_sample(
    network::Flux.Chain,
    input_point::AbstractArray{<:Real},
    true_index::Int,
    perturbation_radius::Real,
    main_algorithm, # constructor
    solve_options::AbstractOptions
)::Dict
    res = Dict()

    # Predicted index
    res[:PredictedIndex] = input_point |> network |> vec |> argmax
    
    # True index
    res[:TrueIndex] = true_index

    # Algorithm infos
    res[:MainAlgorithm] = main_algorithm
    res[:AlgorithmOptions] = solve_options

    # Get the algorithm
    algorithm = main_algorithm{solve_options.u_type}(network, input_point, true_index, perturbation_radius, solve_options)

    #CPUtic()

    # Run the algorithm
    # If the original input is adversarial, it is not necessary to run the verification
    f_core_ops = () -> begin
        if res[:TrueIndex] != res[:PredictedIndex]
            res[:Status] = Adversarial
        else
            init_bounding_algorithm!(algorithm)
            model_network_layers!(algorithm)
            decide_label_uniformity!(algorithm)
        end
    end

    # Mesure CPU time
    #solve_time = CPUtoq()

    solve_time = @elapsed f_core_ops()

    res = merge(res, algorithm.results)

    res[:SolveTime] = solve_time
    res[:Radius] = perturbation_radius
    res[:VerifiedPrecision] = solve_options.u_type

    return res
end


"""
    run_on_batch(main_algorithm, network, dataset, output_indices, perturbation_radius;
                 network_id = "", solve_options, output_path = "./Results")

Run [`run_on_sample`](@ref) on every sample of `dataset`, a `W × H × C × N` array, with the correct classes `output_indices` (1-based), and
append one CSV row per sample to

    <output_path>/<network_id>/<id>.csv

where `<id>` is `IBP`, `Symbolic`, `FPSoundIBP` or `FPSoundSymbolic` (the `FPSound` prefix is
used when `solve_options.fp_sound` is `true`). The settings `over_safe` and `under_safe` do
not appear in the file name, so use a separate `output_path` for each setting.

Existing files are appended to, not overwritten. The columns are described in
[`save_to_disk`](@ref). `output_path` is created with `mkdir`, so its parent directory must
exist.
"""
function run_on_batch(
    main_algorithm, # constructor
    network::Flux.Chain,
    dataset::AbstractArray{<:Real, 4},
    output_indices::AbstractVector{<:Int},
    perturbation_radius::Real;
    # Keyword arguments
    network_id::AbstractString = "",
    solve_options::AbstractOptions,
    output_path::AbstractString = "./Results"
)
    if !isdir(output_path) mkdir(output_path) end
    if !isdir(joinpath(output_path, network_id)) mkdir(joinpath(output_path, network_id)) end

    input_samples = size(dataset, 4)
    
    for input_idx in 1:input_samples
        println("Processing sample $input_idx / $input_samples ...")
        res = run_on_sample(
            network,
            dataset[:, :, :, input_idx:input_idx],
            output_indices[input_idx],
            perturbation_radius,
            main_algorithm,
            solve_options
        )

        out_point = dataset[:, :, :, input_idx:input_idx] |> network |> vec
        
        if haskey(res, :OutputBounds)
            println("Output bounds: $(res[:OutputBounds])")
            println("Output point: $out_point")
        end

        # timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        alg_id = get_alg_id(main_algorithm, solve_options.fp_sound)

        output_file = joinpath(output_path, network_id, "$alg_id.csv")
        
        save_to_disk(
            input_idx,
            res,
            output_file
        )
    end

end


"""
    save_to_disk(input_idx, res, output_file)

Append the result `res` of [`run_on_sample`](@ref) as one row to the CSV file
`output_file`; the header is written when the file is created. Columns:

| Column | Meaning |
|:--- |:--- |
| `SampleIndex` | `input_idx`, the 1-based position of the sample in the dataset |
| `PredictedIndex`, `TrueIndex` | predicted and correct class (1-based) |
| `Status` | the [`Decision`](@ref) |
| `Radius` | perturbation radius |
| `SolveTime` | verification time in seconds |
| `MeanOutputDiameter`, `MaxOutputDiameter`, `MinOutputDiameter` | mean, maximum and minimum width of the output bounds over the output neurons; `-1` if the verifier was not run |
| `VerifiedPrecision` | `u_type` |
| `UnstableNodes` | number of unstable ReLU neurons; `-1` if the verifier was not run |
"""
function save_to_disk(
    input_idx::Int,
    res::Dict,
    output_file::AbstractString
)
    header = ["SampleIndex","PredictedIndex","TrueIndex","Status", "Radius", "SolveTime", "MeanOutputDiameter", "MaxOutputDiameter", "MinOutputDiameter", "VerifiedPrecision", "UnstableNodes"]
    
    if haskey(res, :OutputBounds)
        out_diams = diam.(res[:OutputBounds])
    else
        # Just a placeholder value to keep the csv format consistent
        out_diams = -1
    end

    record = [
        input_idx,
        res[:PredictedIndex],
        res[:TrueIndex],
        res[:Status],
        res[:Radius],
        res[:SolveTime],
        mean(out_diams),
        maximum(out_diams),
        minimum(out_diams),
        res[:VerifiedPrecision],
        get(res, :UnstableNodes, -1)
    ]

    write_header = !isfile(output_file)

    header = join(header, ",")
    record = join(string.(record), ",")


    open(output_file, "a") do io
        if write_header
            write(io, header, "\n")
        end
        write(io, record, "\n")
    end
    
    return
end


"""
    get_alg_id(algorithm, fp_sound::Bool)

Return a string identifying the algorithm based on its type and floating-point soundness.
"""
function get_alg_id(algorithm, fp_sound::Bool)
    alg_name = algorithm == FPSoundIBP.Algorithm ? "IBP" : "Symbolic"
    prefix = fp_sound ? "FPSound" : ""

    return prefix * alg_name
end
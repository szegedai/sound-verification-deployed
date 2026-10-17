## Towards Sound Verification of Deployed Neural Networks [(NeurIPS 2026)](https://github.com/szegedai/sound-verification-deployed)

Code for the paper **Towards Sound Verification of Deployed Neural Networks** (Attila Szász, Balázs Bánhelyi, Márk Jelasity).

Neural network verifiers usually prove properties of the network evaluated in exact real arithmetic. A deployed network, however, is evaluated in floating-point, and the result of a sum depends on the order in which the deployment environment adds its terms (the *expression tree*), which is normally not known in advance. This repository contains two verifiers whose results are **floating-point sound**: if they report a network as robust, it is robust for **every** expression tree and any rounding mode. Possible overflows are detected, and underflow (also in flush-to-zero mode) is handled.

- **FPSoundIBP**: interval bound propagation.
- **FPSoundSymbolic**: symbolic propagation with interval linear expressions.


## Abstract
Verification methods aim at mathematically proving desirable properties of neural networks, such as robustness to adversarial perturbations. 
A verifier is sound if and only if it never claims that a neural network has the desired property when it does not. 
It was shown recently that none of the currently known verifiers that are claimed to be sound are guaranteed to be sound when considering the deployed version of the verified network. 
Due to this, all the known verifiers are vulnerable to certain backdoor attacks, where an adversarial network
passes verification, but in reality, it exhibits adversarial behavior in specific deployment environments.
So far, it has been suspected that sound verification is prohibitively expensive if we wish to verify all possible executions-including parallel and stochastic ones-in deployment. 
*We show that efficient and practically sound verifiers can be designed using a bound on the so-called backward error.*
Using this bounding technique, we propose two verifiers: one based on interval bound propagation, and one using symbolic propagation.
Both verifiers are proven to remain sound even if the deployment environment randomly selects a valid expression tree (an ordering and parenthesizing of the arithmetic operations)
to compute the network, and even if numeric underflow or overflow occurs in any valid expression tree.
This is especially interesting in the case of symbolic propagation, where the expression tree used by the verifier is completely different from the one used in deployment.
We demonstrate empirically that our techniques introduce only a limited performance overhead while detecting all the known verifier backdoor attacks.

## Installation

1. Install Julia (≥ 1.11).
2. Clone the repository and install the exact package versions recorded in `Manifest.toml`.

The verifier is not a registered package; it is loaded by including its main file:

```julia
include("FPSoundVerification.jl")
using .FPSoundVerification
```

## Usage

### Verifying your own network

The networks are `Flux.Chain`s built from `Dense`, `Conv`, `relu` and `Flux.flatten` layers (plus an optional input normalisation layer, `NormalizeLayer`).

```julia
include("FPSoundVerification.jl")
using .FPSoundVerification

opts = from_kwargs(FPSoundSymbolic.Option;
                   input_range = (0.0, 1.0),  # valid input range, of type u_type
                   u_type      = Float64,     # verified precision = deployment precision
                   fp_sound    = true,        # account for floating-point rounding (-s)
                   over_safe   = true,        # detect possible overflows (-o)
                   under_safe  = false)       # flush-to-zero / denormals-are-zero safety (-u)

# network: Flux.Chain; x: input as a W × H × C × 1 array;
# y: 1-based index of the correct class; 0.1: L∞ perturbation radius
res = run_on_sample(network, x, y, 0.1, FPSoundSymbolic.Algorithm, opts)

res[:Status]        # Safe, Adversarial or PossibleUnderOrOverflow
res[:OutputBounds]  # bounds of the network outputs over the whole input ball
```

Use `FPSoundIBP.Algorithm` with `FPSoundIBP.Option` for interval bound propagation, and `run_on_batch` to verify a whole dataset and write the results to a CSV file.

**Interpreting the result.**

- `Safe`: the network is proven robust on the whole input ball.
  - With `fp_sound = true`, this holds for the deployed network: it is evaluated in precision `u_type`, with any expression tree. Two assumptions remain:
    - **no overflow**: with `over_safe = true` this is checked by the verifier (a possible overflow gives `PossibleUnderOrOverflow` instead of `Safe`); without it, it is assumed;
    - **subnormal numbers are supported**: with `under_safe = true` the result also holds in flush-to-zero / denormals-are-zero mode.
  - With `fp_sound = false`, it holds only in exact (real) arithmetic.
- `Adversarial`: robustness could not be proven (or the unperturbed input is already misclassified). Since the bounds over-approximate, this does not mean that an adversarial example exists.
- `PossibleUnderOrOverflow`: an overflow may occur in some neuron; no robustness claim is made.

### Example script

`run_test.jl` shows how a verification run is started. It verifies a small predefined convolutional network (fixed random weights, 3 classes) on a random dataset. For example, to run FPSoundSymbolic with overflow detection on 10 random inputs:

```bash
julia run_test.jl -v Symbolic -n 10 -p 0.01 -r Float64 -s -o
```

The main arguments are `-v` (`IBP` or `Symbolic`), `-n` (number of inputs), `-p` (L∞ perturbation radius), `-r` (`Float32` or `Float64`), `-s` (floating-point sound verification), `-o` (overflow check), `-u` (underflow safety for flush-to-zero environments) and `-d` (output folder, default `./Results`). `julia run_test.jl --help` lists all of them. The first run takes a few minutes, mostly for precompiling the packages, and the `SolveTime` of the first input also includes Julia's compilation time.

## Assumptions

- The verified precision (`u_type`) equals the floating-point precision of the deployment, and the network weights are the floating-point numbers used in deployment.
- The deployment computes each neuron as an **inner product with some expression tree**; approximate algorithms (e.g. approximate matrix multiplication) **are not covered**.
- `nμ < 1`, where `n` is the number of terms of a neuron and `μ` the machine epsilon of the deployment precision.

## Numerical results

The CSV files of the experiments of the paper are in `NumResults`:

* `NumResults/AdvNets`: the backdoored networks ([Szász et al., 2025](https://openreview.net/pdf?id=c16m2kUTLZ)) and their base model `wk17a` (MNIST).
* `NumResults/ConvNets`: the convolutional networks `mnist_conv_small`, `mnist_conv_big`, `mnist_conv_super`, `cifar10_conv_small` and `cifar10_conv_big` ([Ferrari et al., 2022](https://files.sri.inf.ethz.ch/website/papers/ferrari2022complete.pdf)), denoted MS, ML, MXL, CS and CL in the paper.

## Supported features

* Algorithms: interval bound propagation (IBP), symbolic propagation
* Layers: Dense, Convolution (`groups = 1`), Flatten, input normalisation
* Activations: ReLU
* Precisions: `Float32`, `Float64`

## Citation

```bibtex
@inproceedings{szasz2026sound,
  title     = {Towards Sound Verification of Deployed Neural Networks},
  author    = {Attila Sz{\'a}sz and Bal{\'a}zs B{\'a}nhelyi and M{\'a}rk Jelasity},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}

@inproceedings{nosoundness2025,
    title={No Soundness in the Real World: On the Challenges of the Verification of Deployed Neural Networks},
    author={Attila Sz{\'a}sz and Bal{\'a}zs B{\'a}nhelyi and M{\'a}rk Jelasity},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025},
    url={https://arxiv.org/abs/2506.01054}
}

@inproceedings{zombori2021fooling,
    title={Fooling a Complete Neural Network Verifier},
    author={D{\'a}niel Zombori and Bal{\'a}zs B{\'a}nhelyi and Tibor Csendes and Istv{\'a}n Megyeri and M{\'a}rk Jelasity},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2021},
    url={https://openreview.net/forum?id=4IwieFS44l}
}
```

## Licence

This project is licensed under the Apache 2.0 License; see the `LICENCE` file for details.

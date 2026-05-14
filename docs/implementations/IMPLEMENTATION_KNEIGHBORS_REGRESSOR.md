# Implementation Plan: KNeighborsRegressor

## Overview

Add `KNeighborsRegressor` to the `Neighbors` module — the regression counterpart
to the existing `KNeighborsClassifier`. The regressor mirrors the classifier's
struct layout, constructor, fit, and neighbor-lookup logic, replacing majority-vote
prediction with (optionally weighted) mean of neighbor targets.

### Files to create
- `src/Neighbors/KNeighborsRegressor.jl`

### Files to modify
- `src/Neighbors/Neighbors.jl` — include + export the new file
- `test/runtests.jl` — add regression tests

---

## Section 1: Create `src/Neighbors/KNeighborsRegressor.jl`

Create the file `src/Neighbors/KNeighborsRegressor.jl` with the following
complete contents:

```julia
using Distances
using Statistics

import ...NovaML: AbstractModel

mutable struct KNeighborsRegressor <: AbstractModel
    n_neighbors::Int
    weights::Symbol
    algorithm::Symbol
    leaf_size::Int
    p::Real
    metric::Metric
    metric_params::Union{Dict, Nothing}
    n_jobs::Union{Int, Nothing}

    # Attributes
    n_features_in_::Int
    n_samples_fit_::Int

    # Internal data
    X::Union{Matrix{Float64}, Nothing}
    y::Union{Vector{Float64}, Nothing}

    fitted::Bool

    function KNeighborsRegressor(;
        n_neighbors::Int = 5,
        weights::Symbol = :uniform,
        algorithm::Symbol = :auto,
        leaf_size::Int = 30,
        p::Real = 2,
        metric::Union{String, Metric} = "minkowski",
        metric_params::Union{Dict, Nothing} = nothing,
        n_jobs::Union{Int, Nothing} = nothing
    )
        @assert n_neighbors > 0 "n_neighbors must be positive"
        @assert weights in [:uniform, :distance] "weights must be :uniform or :distance"
        @assert algorithm in [:auto, :brute] "Only :auto and :brute algorithms are currently supported"
        @assert leaf_size > 0 "leaf_size must be positive"
        @assert p > 0 "p must be positive"

        if typeof(metric) == String
            if metric == "minkowski"
                metric = Minkowski(p)
            elseif metric == "euclidean"
                metric = Euclidean()
            elseif metric == "manhattan"
                metric = Cityblock()
            else
                error("Unsupported metric string: $metric")
            end
        end

        new(
            n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs,
            0, 0, nothing, nothing, false
        )
    end
end

function (reg::KNeighborsRegressor)(X::Matrix{Float64}, y::Vector)
    reg.X = X
    reg.y = Float64.(y)
    reg.n_features_in_ = size(X, 2)
    reg.n_samples_fit_ = size(X, 1)

    reg.fitted = true
    return reg
end

function (reg::KNeighborsRegressor)(X::Matrix{Float64})
    if !reg.fitted
        throw(ErrorException("This KNeighborsRegressor instance is not fitted yet. Call with training data before using it for predictions."))
    end

    n_samples = size(X, 1)
    predictions = Vector{Float64}(undef, n_samples)

    for i in 1:n_samples
        distances = [evaluate(reg.metric, X[i, :], reg.X[j, :]) for j in 1:reg.n_samples_fit_]
        sorted_indices = sortperm(distances)
        neighbor_indices = sorted_indices[1:reg.n_neighbors]
        neighbor_distances = distances[neighbor_indices]
        neighbor_targets = reg.y[neighbor_indices]

        if reg.weights == :uniform
            predictions[i] = mean(neighbor_targets)
        else # distance weighting
            w = 1.0 ./ (neighbor_distances .+ eps())
            predictions[i] = sum(w .* neighbor_targets) / sum(w)
        end
    end

    return predictions
end

function Base.show(io::IO, reg::KNeighborsRegressor)
    print(io, "KNeighborsRegressor(n_neighbors=$(reg.n_neighbors), ",
        "weights=$(reg.weights), algorithm=$(reg.algorithm), ",
        "leaf_size=$(reg.leaf_size), p=$(reg.p), metric=$(reg.metric), ",
        "metric_params=$(reg.metric_params), n_jobs=$(reg.n_jobs), fitted=$(reg.fitted))")
end
```

### Design decisions vs. `KNeighborsClassifier`

| Aspect | Classifier | Regressor |
|--------|-----------|-----------|
| Fitted attr `classes_` | `Vector{Any}` of unique labels | **Removed** — not meaningful for regression |
| Fitted attr `outputs_2d_` | Boolean | **Removed** — not needed |
| `y` type | `Vector` (any eltype) | `Vector{Float64}` — regression targets are numeric |
| predict (uniform) | `mode()` — majority vote | `mean()` — arithmetic mean |
| predict (distance) | `sample()` with weights | weighted mean: `Σ(wᵢyᵢ) / Σ(wᵢ)` |
| `type=:probs` kwarg | Returns class probabilities | **Not supported** — no classes to have probabilities over |
| Imports | `StatsBase: mode, sample, Weights` | No StatsBase import — `mean` comes from `Statistics`; weighted mean is computed manually |

### Verification for Section 1

Run from the project root:

```bash
julia -e '
    using Pkg; Pkg.activate(".")
    include("src/Neighbors/KNeighborsRegressor.jl")
    println("File parses successfully")
'
```

This verifies the file is syntactically valid Julia. The include will fail at the
`import ...NovaML` line when run standalone — that is expected and confirms the
relative import is present. A parse-level check is:

```bash
julia -e 'Meta.parse(read("src/Neighbors/KNeighborsRegressor.jl", String)); println("Parse OK")'
```

---

## Section 2: Wire into the Neighbors module

### Modify `src/Neighbors/Neighbors.jl`

Append two lines before the closing `end` statement, after the existing
`KNeighborsClassifier` block:

```diff
 module Neighbors

 include("KNeighborsClassifier.jl")
 export KNeighborsClassifier

+include("KNeighborsRegressor.jl")
+export KNeighborsRegressor
+
 end
```

### Verification for Section 2

```bash
julia --project=. -e '
    using NovaML
    reg = NovaML.Neighbors.KNeighborsRegressor(n_neighbors=3)
    println(reg)
    println("Module wiring OK")
'
```

Expected output includes the `KNeighborsRegressor(...)` show string and
`"Module wiring OK"`.

Also verify that the existing classifier still works:

```bash
julia --project=. -e '
    using NovaML
    clf = NovaML.Neighbors.KNeighborsClassifier(n_neighbors=3)
    println(clf)
    println("Classifier still works")
'
```

---

## Section 3: Add tests

### Modify `test/runtests.jl`

Replace the entire contents of `test/runtests.jl` with:

```julia
using NovaML
using Test

@testset "NovaML.jl" begin

    @testset "KNeighborsRegressor" begin
        using NovaML.Neighbors: KNeighborsRegressor

        @testset "Constructor defaults" begin
            reg = KNeighborsRegressor()
            @test reg.n_neighbors == 5
            @test reg.weights == :uniform
            @test reg.algorithm == :auto
            @test reg.leaf_size == 30
            @test reg.p == 2
            @test reg.fitted == false
            @test reg.X === nothing
            @test reg.y === nothing
        end

        @testset "Constructor custom params" begin
            reg = KNeighborsRegressor(n_neighbors=3, weights=:distance, p=1, metric="manhattan")
            @test reg.n_neighbors == 3
            @test reg.weights == :distance
        end

        @testset "Constructor validation" begin
            @test_throws AssertionError KNeighborsRegressor(n_neighbors=0)
            @test_throws AssertionError KNeighborsRegressor(n_neighbors=-1)
            @test_throws AssertionError KNeighborsRegressor(weights=:invalid)
            @test_throws AssertionError KNeighborsRegressor(leaf_size=0)
            @test_throws AssertionError KNeighborsRegressor(p=0)
        end

        @testset "Fit" begin
            reg = KNeighborsRegressor(n_neighbors=2)
            X_train = [1.0 0.0; 2.0 0.0; 3.0 0.0; 4.0 0.0]
            y_train = [1.0, 2.0, 3.0, 4.0]
            reg(X_train, y_train)

            @test reg.fitted == true
            @test reg.n_features_in_ == 2
            @test reg.n_samples_fit_ == 4
            @test reg.X == X_train
            @test reg.y == y_train
        end

        @testset "Predict not fitted" begin
            reg = KNeighborsRegressor()
            @test_throws ErrorException reg(Float64[1.0 0.0])
        end

        @testset "Predict uniform weights" begin
            reg = KNeighborsRegressor(n_neighbors=2)
            X_train = [1.0 0.0; 2.0 0.0; 3.0 0.0; 4.0 0.0]
            y_train = [10.0, 20.0, 30.0, 40.0]
            reg(X_train, y_train)

            # Query point [1.5, 0.0] — two nearest are [1,0] (y=10) and [2,0] (y=20)
            preds = reg(Float64[1.5 0.0])
            @test length(preds) == 1
            @test preds[1] ≈ 15.0  # mean(10, 20)

            # Query point [3.5, 0.0] — two nearest are [3,0] (y=30) and [4,0] (y=40)
            preds = reg(Float64[3.5 0.0])
            @test preds[1] ≈ 35.0  # mean(30, 40)
        end

        @testset "Predict distance weights" begin
            reg = KNeighborsRegressor(n_neighbors=2, weights=:distance)
            X_train = [0.0 0.0; 10.0 0.0]
            y_train = [0.0, 10.0]
            reg(X_train, y_train)

            # Query point [1.0, 0.0]: dist to [0,0]=1, dist to [10,0]=9
            # w1 = 1/1 = 1, w2 = 1/9 ≈ 0.111
            # pred = (1*0 + 0.111*10) / (1 + 0.111) = 1.111/1.111 ≈ 1.0
            # (Using exact: w1=1/(1+eps), w2=1/(9+eps), result ≈ 1.0)
            preds = reg(Float64[1.0 0.0])
            @test preds[1] ≈ 1.0 atol=0.1

            # Query point [9.0, 0.0]: dist to [0,0]=9, dist to [10,0]=1
            # Closer to 10.0 → prediction should be near 9.0
            preds = reg(Float64[9.0 0.0])
            @test preds[1] ≈ 9.0 atol=0.1
        end

        @testset "Predict distance weights exact match" begin
            reg = KNeighborsRegressor(n_neighbors=2, weights=:distance)
            X_train = [0.0 0.0; 10.0 0.0]
            y_train = [5.0, 50.0]
            reg(X_train, y_train)

            # Query exactly on a training point: distance=0, w=1/eps() dominates
            preds = reg(Float64[0.0 0.0])
            @test preds[1] ≈ 5.0 atol=1e-10
        end

        @testset "Predict multiple samples" begin
            reg = KNeighborsRegressor(n_neighbors=1)
            X_train = [0.0 0.0; 1.0 0.0; 2.0 0.0]
            y_train = [100.0, 200.0, 300.0]
            reg(X_train, y_train)

            X_test = [0.1 0.0; 0.9 0.0; 2.1 0.0]
            preds = reg(X_test)
            @test length(preds) == 3
            @test preds[1] ≈ 100.0  # nearest to [0,0]
            @test preds[2] ≈ 200.0  # nearest to [1,0]
            @test preds[3] ≈ 300.0  # nearest to [2,0]
        end

        @testset "Fit returns self" begin
            reg = KNeighborsRegressor()
            X_train = [1.0 0.0; 2.0 0.0]
            y_train = [1.0, 2.0]
            result = reg(X_train, y_train)
            @test result === reg
        end

        @testset "Integer y converted to Float64" begin
            reg = KNeighborsRegressor(n_neighbors=1)
            X_train = [1.0 0.0; 2.0 0.0]
            y_train = [1, 2]  # integers
            reg(X_train, y_train)
            @test eltype(reg.y) == Float64
        end

        @testset "show method" begin
            reg = KNeighborsRegressor(n_neighbors=3)
            buf = IOBuffer()
            show(buf, reg)
            s = String(take!(buf))
            @test occursin("KNeighborsRegressor", s)
            @test occursin("n_neighbors=3", s)
            @test occursin("fitted=false", s)
        end
    end

end
```

### Verification for Section 3

```bash
cd /Users/ilker/Documents/NovaML.jl && julia --project=. -e 'using Pkg; Pkg.test()'
```

All tests should pass. Expected output ends with `Test Summary` showing all
`KNeighborsRegressor` subtests as passed.

---

## Testing Strategy

All testing is automated (categories 1 and 2 only):

| Test | What it verifies |
|------|-----------------|
| Constructor defaults | All default parameter values match specification |
| Constructor custom params | Non-default values are stored correctly |
| Constructor validation | Invalid inputs raise `AssertionError` |
| Fit | Training data stored, attributes set, `fitted=true` |
| Predict not fitted | Error thrown when predicting before fit |
| Predict uniform weights | Arithmetic mean of k-nearest targets (deterministic, hand-computed) |
| Predict distance weights | Inverse-distance-weighted mean (deterministic, hand-computed) |
| Predict distance weights exact match | Query on training point (distance=0) returns that point's target |
| Predict multiple samples | Batch prediction returns correct vector length and values |
| Fit returns self | Callable fit returns the regressor instance |
| Integer y converted | Numeric coercion of integer targets to Float64 |
| show method | String representation includes struct name and params |

All expected values are hand-computed from simple 1D-along-axis geometries where
the nearest neighbors are unambiguous.

---

## Operator Validation Checklist

This section is owned by the human operator, not by the creator
subprocess. Maestro's implementation reviewer MUST NOT escalate a
section solely because items in this checklist read
`pending operator validation`. Maestro's final reviewer MUST allow
`Implementation Complete: YES` when all creator-performable work
and all automated checks have passed, even if items here remain
`pending operator validation`.

Each row has one of three states:

- `pending operator validation` — the action has not been performed
  yet. This is the default state. It is HONEST and ALLOWED. Do not
  rewrite it to `passed` unless the operator has actually performed
  and recorded the action.
- `passed` — the operator performed the action and recorded the
  evidence (date, host, screenshot path, log path, or commit SHA).
  The evidence MUST be specific. A bare `passed` with no captured
  artifact is treated as fabricated.
- `waived by operator` — the operator decided the action is not
  required for this slice and recorded the reason inline.

### Manual smoke

- [ ] `pending operator validation` — `julia --project=.` REPL walkthrough:
  instantiate `KNeighborsRegressor`, fit on a small dataset, predict, verify
  output looks reasonable. Evidence: `<terminal log or note when run>`.
- [ ] `pending operator validation` — Verify `KNeighborsClassifier` is
  unaffected: fit + predict with the classifier after the new code is added.
  Evidence: `<terminal log or note when run>`.

### External / live-host checks

- (none required for this slice)

### Commit / release hygiene

- [ ] `pending operator validation` — Commit the final shipped state
  of the files listed in this slice (`src/Neighbors/KNeighborsRegressor.jl`,
  `src/Neighbors/Neighbors.jl`, `test/runtests.jl`).
  Evidence: `<commit SHA when landed>`.
- [ ] `pending operator validation` — Tag a new patch version if appropriate
  (currently `v0.3.4`). Evidence: `<tag name when created>`.

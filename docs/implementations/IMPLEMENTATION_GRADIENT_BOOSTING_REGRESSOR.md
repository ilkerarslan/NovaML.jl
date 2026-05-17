# Implementation Plan: GradientBoostingRegressor

## Overview

Add `GradientBoostingRegressor` to the `Ensemble` module — the regression
counterpart to the existing `GradientBoostingClassifier`. The regressor mirrors
the classifier's struct layout and boosting loop, but replaces classification
losses/predictions with regression losses (`squared_error`, `absolute_error`,
`huber`) and numeric predictions (initial mean + sum of learning_rate-scaled
tree predictions).

### Intentional API differences from `GradientBoostingClassifier`

- **`init`** — accepts only `String` (`"zero"`) or `Nothing` (defaults to
  `mean(y)`). The classifier also accepts `AbstractModel` for custom initial
  estimators; the regressor omits this because a scalar initial prediction
  (`mean` or zero) covers standard regression use-cases and avoids the
  per-sample initial-estimator machinery the classifier requires for
  multi-class probability estimation.
- **`min_samples_leaf` (float)** — validated to `(0, 0.5]`, not `(0, 1.0)`.
  Values above 0.5 make splits mathematically impossible (both children of a
  binary split would need > 50 % of samples). scikit-learn silently accepts
  the wider range; NovaML rejects it at construction time to surface the
  misconfiguration early.

### Files to create
- `src/Ensemble/GradientBoostingRegressor.jl`

### Files to modify
- `src/Ensemble/Ensemble.jl` — include + export the new file
- `test/runtests.jl` — add regression tests

---

## Section 1: Create `src/Ensemble/GradientBoostingRegressor.jl`

Create the file `src/Ensemble/GradientBoostingRegressor.jl` with the following
complete contents:

```julia
using Random
using Statistics

import ...NovaML: AbstractModel
import ..Tree: DecisionTreeRegressor

mutable struct GradientBoostingRegressor <: AbstractModel
    loss::String
    learning_rate::Float64
    n_estimators::Int
    subsample::Float64
    criterion::String
    min_samples_split::Union{Int, Float64}
    min_samples_leaf::Union{Int, Float64}
    min_weight_fraction_leaf::Float64
    max_depth::Union{Int, Nothing}
    min_impurity_decrease::Float64
    init::Union{String, Nothing}
    random_state::Union{Int, Nothing}
    max_features::Union{Int, Float64, String, Nothing}
    verbose::Int
    max_leaf_nodes::Union{Int, Nothing}
    warm_start::Bool
    validation_fraction::Float64
    n_iter_no_change::Union{Int, Nothing}
    tol::Float64
    ccp_alpha::Float64
    alpha::Float64  # quantile for huber loss

    # Fitted attributes
    estimators_::Vector{DecisionTreeRegressor}
    init_prediction_::Float64
    feature_importances_::Union{Vector{Float64}, Nothing}
    train_score_::Vector{Float64}
    n_estimators_::Int
    fitted::Bool

    function GradientBoostingRegressor(;
        loss::String="squared_error",
        learning_rate::Float64=0.1,
        n_estimators::Int=100,
        subsample::Float64=1.0,
        criterion::String="friedman_mse",
        min_samples_split::Union{Int, Float64}=2,
        min_samples_leaf::Union{Int, Float64}=1,
        min_weight_fraction_leaf::Float64=0.0,
        max_depth::Union{Int, Nothing}=3,
        min_impurity_decrease::Float64=0.0,
        init::Union{String, Nothing}=nothing,
        random_state::Union{Int, Nothing}=nothing,
        max_features::Union{Int, Float64, String, Nothing}=nothing,
        verbose::Int=0,
        max_leaf_nodes::Union{Int, Nothing}=nothing,
        warm_start::Bool=false,
        validation_fraction::Float64=0.1,
        n_iter_no_change::Union{Int, Nothing}=nothing,
        tol::Float64=1e-4,
        ccp_alpha::Float64=0.0,
        alpha::Float64=0.9
    )
        loss in ("squared_error", "absolute_error", "huber") ||
            throw(ArgumentError("loss must be one of: squared_error, absolute_error, huber. Got: $loss"))
        init === nothing || init == "zero" ||
            throw(ArgumentError("init must be nothing or \"zero\". Got: $init"))
        0.0 < learning_rate <= 1.0 ||
            throw(ArgumentError("learning_rate must be in (0, 1]. Got: $learning_rate"))
        n_estimators > 0 ||
            throw(ArgumentError("n_estimators must be positive. Got: $n_estimators"))
        0.0 < subsample <= 1.0 ||
            throw(ArgumentError("subsample must be in (0, 1]. Got: $subsample"))
        0.0 < validation_fraction < 1.0 ||
            throw(ArgumentError("validation_fraction must be in (0, 1). Got: $validation_fraction"))
        0.0 < alpha < 1.0 ||
            throw(ArgumentError("alpha must be in (0, 1). Got: $alpha"))
        if min_samples_split isa Float64
            0.0 < min_samples_split <= 1.0 ||
                throw(ArgumentError("min_samples_split as float must be in (0, 1]. Got: $min_samples_split"))
        else
            min_samples_split >= 2 ||
                throw(ArgumentError("min_samples_split as int must be >= 2. Got: $min_samples_split"))
        end
        if min_samples_leaf isa Float64
            0.0 < min_samples_leaf <= 0.5 ||
                throw(ArgumentError("min_samples_leaf as float must be in (0, 0.5]. Got: $min_samples_leaf"))
        else
            min_samples_leaf >= 1 ||
                throw(ArgumentError("min_samples_leaf as int must be >= 1. Got: $min_samples_leaf"))
        end
        new(
            loss, learning_rate, n_estimators, subsample, criterion,
            min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_depth, min_impurity_decrease, init, random_state,
            max_features, verbose, max_leaf_nodes, warm_start,
            validation_fraction, n_iter_no_change, tol, ccp_alpha, alpha,
            DecisionTreeRegressor[], 0.0, nothing, Float64[], 0, false
        )
    end
end

function (gbr::GradientBoostingRegressor)(X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)

    if gbr.random_state !== nothing
        Random.seed!(gbr.random_state)
    end

    # Split for early stopping validation
    if gbr.n_iter_no_change !== nothing
        n_samples >= 2 ||
            throw(ArgumentError("n_iter_no_change requires at least 2 samples. Got: $n_samples"))
        n_val = clamp(round(Int, n_samples * gbr.validation_fraction), 1, n_samples - 1)
        indices = randperm(n_samples)
        val_indices = indices[1:n_val]
        train_indices = indices[n_val+1:end]
        X_train, y_train = X[train_indices, :], y[train_indices]
        X_val, y_val = X[val_indices, :], y[val_indices]
    else
        X_train, y_train = X, y
        X_val, y_val = nothing, nothing
    end

    n_train = size(X_train, 1)

    # Resolve min_samples_split/min_samples_leaf: float means fraction of n_train
    eff_min_samples_split = gbr.min_samples_split isa Float64 ?
        max(2, ceil(Int, n_train * gbr.min_samples_split)) :
        gbr.min_samples_split
    eff_min_samples_leaf = gbr.min_samples_leaf isa Float64 ?
        max(1, ceil(Int, n_train * gbr.min_samples_leaf)) :
        gbr.min_samples_leaf

    # Initialize estimators
    if !gbr.warm_start || isempty(gbr.estimators_)
        gbr.estimators_ = DecisionTreeRegressor[]
        gbr.train_score_ = Float64[]
    end

    # Determine how many new trees to add (warm_start continues from existing count)
    if gbr.warm_start && !isempty(gbr.estimators_)
        n_existing = length(gbr.estimators_)
        if n_existing > gbr.n_estimators
            throw(ArgumentError(
                "n_estimators=$(gbr.n_estimators) must be >= length(estimators_)=$n_existing when warm_start=true"
            ))
        elseif n_existing == gbr.n_estimators
            gbr.n_estimators_ = n_existing
            gbr.feature_importances_ = _compute_feature_importances_reg(gbr)
            gbr.fitted = true
            return gbr
        end
        n_new_estimators = gbr.n_estimators - n_existing
    else
        n_new_estimators = gbr.n_estimators
    end

    # Initialize prediction
    if gbr.init == "zero"
        gbr.init_prediction_ = 0.0
    else
        gbr.init_prediction_ = mean(y_train)
    end

    # Current predictions — include prior estimators if warm_start
    y_pred = fill(gbr.init_prediction_, n_train)
    if gbr.warm_start && !isempty(gbr.estimators_)
        for tree in gbr.estimators_
            y_pred .+= gbr.learning_rate .* tree(X_train)
        end
    end
    if X_val !== nothing
        y_val_pred = fill(gbr.init_prediction_, length(y_val))
        if gbr.warm_start && !isempty(gbr.estimators_)
            for tree in gbr.estimators_
                y_val_pred .+= gbr.learning_rate .* tree(X_val)
            end
        end
    end

    # Compute huber delta if needed
    huber_delta = 0.0
    if gbr.loss == "huber"
        residuals = y_train .- y_pred
        huber_delta = quantile(abs.(residuals), gbr.alpha)
    end

    # Track best validation score for early stopping
    best_val_score = Inf
    no_improvement_count = 0

    # Main boosting loop — only add trees needed to reach n_estimators
    for i in 1:n_new_estimators
        # Compute negative gradient (pseudo-residuals)
        negative_gradient = _compute_negative_gradient_reg(y_train, y_pred, gbr.loss, huber_delta)

        # Fit a regression tree to the negative gradient
        tree = DecisionTreeRegressor(
            max_depth=gbr.max_depth,
            criterion=gbr.criterion,
            min_samples_split=eff_min_samples_split,
            min_samples_leaf=eff_min_samples_leaf,
            min_weight_fraction_leaf=gbr.min_weight_fraction_leaf,
            max_features=gbr.max_features,
            max_leaf_nodes=gbr.max_leaf_nodes,
            min_impurity_decrease=gbr.min_impurity_decrease,
            ccp_alpha=gbr.ccp_alpha,
            random_state=gbr.random_state !== nothing ? rand(1:10000) : nothing
        )

        # Subsample if needed
        if gbr.subsample < 1.0
            sample_indices = rand(1:n_train, max(1, round(Int, n_train * gbr.subsample)))
            X_subsample = X_train[sample_indices, :]
            neg_grad_subsample = negative_gradient[sample_indices]
        else
            X_subsample = X_train
            neg_grad_subsample = negative_gradient
        end

        tree(X_subsample, neg_grad_subsample)
        push!(gbr.estimators_, tree)

        # Update predictions
        y_pred .+= gbr.learning_rate .* tree(X_train)

        # Update huber delta
        if gbr.loss == "huber"
            residuals = y_train .- y_pred
            huber_delta = quantile(abs.(residuals), gbr.alpha)
        end

        # Store train score
        push!(gbr.train_score_, _compute_loss_reg(y_train, y_pred, gbr.loss, huber_delta))

        # Early stopping check on validation set
        if gbr.n_iter_no_change !== nothing
            y_val_pred .+= gbr.learning_rate .* tree(X_val)
            val_score = _compute_loss_reg(y_val, y_val_pred, gbr.loss, huber_delta)

            if val_score < best_val_score - gbr.tol
                best_val_score = val_score
                no_improvement_count = 0
            else
                no_improvement_count += 1
            end

            if no_improvement_count >= gbr.n_iter_no_change
                break
            end
        end
    end

    gbr.n_estimators_ = length(gbr.estimators_)
    gbr.feature_importances_ = _compute_feature_importances_reg(gbr)
    gbr.fitted = true
    return gbr
end

function (gbr::GradientBoostingRegressor)(X::AbstractMatrix)
    if !gbr.fitted
        throw(ErrorException("This GradientBoostingRegressor instance is not fitted yet. Call the model with training data before using it for predictions."))
    end

    predictions = fill(gbr.init_prediction_, size(X, 1))
    for tree in gbr.estimators_
        predictions .+= gbr.learning_rate .* tree(X)
    end
    return predictions
end

function _compute_negative_gradient_reg(y::AbstractVector, y_pred::AbstractVector, loss::String, huber_delta::Float64)
    if loss == "squared_error"
        return y .- y_pred
    elseif loss == "absolute_error"
        return sign.(y .- y_pred)
    elseif loss == "huber"
        residuals = y .- y_pred
        return [abs(r) <= huber_delta ? r : huber_delta * sign(r) for r in residuals]
    else
        throw(ArgumentError("Unsupported loss function: $loss"))
    end
end

function _compute_loss_reg(y::AbstractVector, y_pred::AbstractVector, loss::String, huber_delta::Float64)
    if loss == "squared_error"
        return mean((y .- y_pred).^2)
    elseif loss == "absolute_error"
        return mean(abs.(y .- y_pred))
    elseif loss == "huber"
        residuals = abs.(y .- y_pred)
        return mean([r <= huber_delta ? 0.5 * r^2 : huber_delta * r - 0.5 * huber_delta^2 for r in residuals])
    else
        throw(ArgumentError("Unsupported loss function: $loss"))
    end
end

function _compute_feature_importances_reg(gbr::GradientBoostingRegressor)
    if isempty(gbr.estimators_)
        return nothing
    end

    n_features = gbr.estimators_[1].n_features_
    importances = zeros(n_features)

    for tree in gbr.estimators_
        if tree.feature_importances_ !== nothing && all(isfinite, tree.feature_importances_)
            importances .+= tree.feature_importances_
        end
    end

    total = sum(importances)
    if total > 0 && isfinite(total)
        importances ./= total
    end
    return importances
end

function Base.show(io::IO, gbr::GradientBoostingRegressor)
    println(io, "GradientBoostingRegressor(")
    println(io, "  loss=$(gbr.loss),")
    println(io, "  learning_rate=$(gbr.learning_rate),")
    println(io, "  n_estimators=$(gbr.n_estimators),")
    println(io, "  subsample=$(gbr.subsample),")
    println(io, "  criterion=$(gbr.criterion),")
    println(io, "  min_samples_split=$(gbr.min_samples_split),")
    println(io, "  min_samples_leaf=$(gbr.min_samples_leaf),")
    println(io, "  min_weight_fraction_leaf=$(gbr.min_weight_fraction_leaf),")
    println(io, "  max_depth=$(gbr.max_depth),")
    println(io, "  min_impurity_decrease=$(gbr.min_impurity_decrease),")
    println(io, "  init=$(gbr.init),")
    println(io, "  random_state=$(gbr.random_state),")
    println(io, "  max_features=$(gbr.max_features),")
    println(io, "  verbose=$(gbr.verbose),")
    println(io, "  max_leaf_nodes=$(gbr.max_leaf_nodes),")
    println(io, "  warm_start=$(gbr.warm_start),")
    println(io, "  validation_fraction=$(gbr.validation_fraction),")
    println(io, "  n_iter_no_change=$(gbr.n_iter_no_change),")
    println(io, "  tol=$(gbr.tol),")
    println(io, "  ccp_alpha=$(gbr.ccp_alpha),")
    println(io, "  alpha=$(gbr.alpha),")
    println(io, "  fitted=$(gbr.fitted)")
    print(io, ")")
end
```

---

## Section 2: Update `src/Ensemble/Ensemble.jl`

Apply two additive edits to the existing file (do NOT replace the entire file):

**Edit 1** — Add the include after line 5 (`include("GradientBoostingClassifier.jl")`):

```julia
include("GradientBoostingRegressor.jl")
```

**Edit 2** — Add `GradientBoostingRegressor` to the existing export list. Change:

```julia
export AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, VotingClassifier
```

to:

```julia
export AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, VotingClassifier
```

---

## Section 3: Add tests in `test/runtests.jl`

Append the following test block at the end of `test/runtests.jl`, inside the
outer `@testset "NovaML.jl"` block (before the final `end`):

```julia
    @testset "GradientBoostingRegressor" begin
        using NovaML.Ensemble: GradientBoostingRegressor
        using Random

        @testset "Constructor defaults" begin
            gbr = GradientBoostingRegressor()
            @test gbr.loss == "squared_error"
            @test gbr.learning_rate == 0.1
            @test gbr.n_estimators == 100
            @test gbr.subsample == 1.0
            @test gbr.max_depth == 3
            @test gbr.alpha == 0.9
            @test gbr.fitted == false
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError GradientBoostingRegressor(loss="invalid")
            @test_throws ArgumentError GradientBoostingRegressor(init="bad")
            @test_throws ArgumentError GradientBoostingRegressor(learning_rate=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(learning_rate=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(n_estimators=0)
            @test_throws ArgumentError GradientBoostingRegressor(subsample=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(subsample=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(validation_fraction=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(validation_fraction=1.0)
            @test_throws ArgumentError GradientBoostingRegressor(alpha=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(alpha=1.0)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=1)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0.0)
            # NovaML caps float min_samples_leaf at 0.5 (values > 0.5 make splits impossible)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0.6)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0)
        end

        @testset "Fit and predict — squared_error" begin
            Random.seed!(42)
            X_train = randn(100, 3)
            y_train = 2.0 .* X_train[:, 1] .+ 0.5 .* X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.n_estimators_ == 50
            @test length(gbr.estimators_) == 50
            @test length(gbr.train_score_) == 50

            # Train loss decreases over boosting rounds
            @test gbr.train_score_[end] < gbr.train_score_[1]

            # Predictions are numeric vector
            preds = gbr(X_train)
            @test length(preds) == 100
            @test eltype(preds) <: AbstractFloat

            # Feature importances
            @test gbr.feature_importances_ !== nothing
            @test length(gbr.feature_importances_) == 3
            @test all(gbr.feature_importances_ .>= 0)
        end

        @testset "Fit and predict — absolute_error" begin
            Random.seed!(123)
            X_train = randn(80, 2)
            y_train = 3.0 .* X_train[:, 1] .+ randn(80) .* 0.2

            gbr = GradientBoostingRegressor(
                loss="absolute_error",
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                random_state=123
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.train_score_[end] < gbr.train_score_[1]

            preds = gbr(X_train)
            @test length(preds) == 80
            @test eltype(preds) <: AbstractFloat
        end

        @testset "Fit and predict — huber" begin
            Random.seed!(456)
            X_train = randn(80, 2)
            y_train = 1.5 .* X_train[:, 1] .- 0.5 .* X_train[:, 2] .+ randn(80) .* 0.3

            gbr = GradientBoostingRegressor(
                loss="huber",
                alpha=0.9,
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                random_state=456
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.train_score_[end] < gbr.train_score_[1]

            preds = gbr(X_train)
            @test length(preds) == 80
            @test eltype(preds) <: AbstractFloat
        end

        @testset "Early stopping" begin
            Random.seed!(789)
            X_train = randn(200, 2)
            y_train = X_train[:, 1] .+ randn(200) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.5,
                max_depth=3,
                n_iter_no_change=10,
                tol=1e-4,
                validation_fraction=0.2,
                random_state=789
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            # Should stop before reaching 500 estimators
            @test gbr.n_estimators_ < 500
        end

        @testset "Predict not fitted" begin
            gbr = GradientBoostingRegressor()
            @test_throws ErrorException gbr(randn(5, 2))
        end

        @testset "Subsample" begin
            Random.seed!(101)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                subsample=0.8,
                learning_rate=0.1,
                max_depth=3,
                random_state=101
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            preds = gbr(X_train)
            @test length(preds) == 100
        end

        @testset "Warm start continuation" begin
            Random.seed!(202)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=202
            )
            gbr(X_train, y_train)
            @test gbr.fitted == true
            @test length(gbr.estimators_) == 30

            # Continue training with more estimators
            gbr.n_estimators = 60
            gbr(X_train, y_train)
            @test length(gbr.estimators_) == 60
            @test gbr.n_estimators_ == 60
            @test length(gbr.train_score_) == 60
        end

        @testset "Warm start — lowering n_estimators errors" begin
            Random.seed!(303)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=303
            )
            gbr(X_train, y_train)
            @test length(gbr.estimators_) == 30

            # Lowering n_estimators with warm_start should throw
            gbr.n_estimators = 20
            @test_throws ArgumentError gbr(X_train, y_train)
        end

        @testset "Warm start — same n_estimators is no-op" begin
            Random.seed!(404)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=404
            )
            gbr(X_train, y_train)
            preds1 = gbr(X_train)

            # Re-fit with same n_estimators — should keep same estimators
            gbr(X_train, y_train)
            preds2 = gbr(X_train)
            @test length(gbr.estimators_) == 30
            @test preds1 == preds2
        end
    end
```

---

## Section 4: Verification

Run the following commands to verify the implementation compiles and tests pass:

**Command 1 — Package precompilation:**
```bash
cd /Users/ilker/Documents/NovaML.jl && julia --project=. -e 'using Pkg; Pkg.instantiate(); using NovaML; println("Precompilation OK")'
```

Expected: prints `Precompilation OK` without errors.

**Command 2 — Smoke-test the new type is exported:**
```bash
cd /Users/ilker/Documents/NovaML.jl && julia --project=. -e '
using NovaML.Ensemble
gbr = GradientBoostingRegressor()
@assert :GradientBoostingRegressor in names(NovaML.Ensemble)
println("Export OK: ", typeof(gbr))
'
```

Expected: prints `Export OK: GradientBoostingRegressor`. The `names()` assertion
confirms the symbol is in the module's public export list, not merely accessible
via explicit import.

**Command 3 — Run full test suite:**
```bash
cd /Users/ilker/Documents/NovaML.jl && julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all test sets pass including the new `GradientBoostingRegressor` tests.

---

## Testing Strategy

### Automated (Category 2 — run by creator)

| Test | Validates |
|------|-----------|
| Constructor defaults | All default hyperparameters set correctly |
| Constructor validation | Invalid `loss`, `init`, `learning_rate`, `n_estimators`, `subsample`, `validation_fraction`, `alpha`, `min_samples_split` (int and float bounds), `min_samples_leaf` (int and float bounds) all throw `ArgumentError` |
| Fit + predict (squared_error) | Model fits, train loss decreases, predictions are `Float64` vectors |
| Fit + predict (absolute_error) | MAE loss variant works end-to-end |
| Fit + predict (huber) | Huber loss variant works end-to-end |
| Early stopping | Model stops before `n_estimators` when validation loss plateaus |
| Predict not fitted | Throws `ErrorException` on unfitted model |
| Subsample | Stochastic gradient boosting with `subsample < 1.0` works |
| Feature importances | Non-null, correct length, non-negative |
| Warm start continuation | Second fit with higher `n_estimators` adds trees correctly |
| Warm start — lowering n_estimators errors | Throws `ArgumentError` when `n_estimators < len(estimators_)` |
| Warm start — same n_estimators is no-op | Re-fit with same count preserves predictions |

All tests run via `Pkg.test()` — no manual intervention required.

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

- [ ] `pending operator validation` — Open a Julia REPL, run
  `using NovaML.Ensemble: GradientBoostingRegressor` and manually
  exercise fit/predict on a small dataset to confirm output looks
  reasonable. Evidence: `<date and note when run>`.

### External / live-host checks

- [ ] `pending operator validation` — N/A for this slice (no
  external services involved). Evidence: `waived — pure library code`.

### Commit / release hygiene

- [ ] `pending operator validation` — Commit the final shipped state
  of the files listed in this slice (`src/Ensemble/GradientBoostingRegressor.jl`,
  `src/Ensemble/Ensemble.jl`, `test/runtests.jl`).
  Evidence: `<commit SHA when landed>`.
- [ ] `pending operator validation` — Update package version in
  `Project.toml` if required by project release conventions.
  Evidence: `<commit SHA or waiver>`.

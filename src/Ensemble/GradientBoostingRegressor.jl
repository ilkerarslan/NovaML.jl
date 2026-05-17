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

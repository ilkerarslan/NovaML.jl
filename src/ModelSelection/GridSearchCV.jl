using Statistics: mean
using ...Metrics: accuracy_score
using ...ModelSelection: cross_val_score
using ...NovaML: _generate_param_combinations, _param_product, _set_params!
    
mutable struct GridSearchCV
    estimator
    param_grid::Vector{Vector{Any}}
    scoring::Function
    cv::Int
    refit::Bool
    best_score::Float64
    best_params::Vector{Any}
    cv_results::Dict

    function GridSearchCV(;
        estimator,
        param_grid,
        scoring=accuracy_score,
        cv=5,
        refit=true
    )
        new(estimator, param_grid, scoring, cv, refit, -Inf, Vector{Any}(), Dict())
    end
end

function (gs::GridSearchCV)(X, y)
    all_params = _generate_param_combinations(gs.param_grid)
    n_combinations = length(all_params)
    
    scores = zeros(n_combinations)
    println("Evaluating $n_combinations combinations.")
    Threads.@threads for i in 1:n_combinations            
        params = all_params[i]
        model = deepcopy(gs.estimator)
        _set_params!(params)
        cv_scores = cross_val_score(model, X, y; cv=gs.cv, scoring=gs.scoring)
        scores[i] = mean(cv_scores)
    end
    
    best_idx = argmax(scores)
    gs.best_score = scores[best_idx]
    gs.best_params = all_params[best_idx]
    
    gs.cv_results = Dict(
        "mean_test_score" => scores,
        "params" => all_params
    )
    
    if gs.refit            
        _set_params!(gs.best_params)
        gs.estimator(X, y)
    end
    
    return gs
end
using Statistics: mean
using Random: shuffle!
using ...Metrics: accuracy_score
using ...ModelSelection: cross_val_score
using ...NovaML: _generate_param_combinations, _param_product, _set_params!

mutable struct RandomSearchCV
    estimator
    param_grid::Vector{Vector{Any}}
    scoring::Function
    cv::Int
    refit::Bool
    n_iter::Int
    random_state::Union{Int, Nothing}
    best_score::Float64
    best_params::Vector{Any}
    cv_results::Dict

    function RandomSearchCV(;
        estimator,
        param_grid,
        scoring=accuracy_score,
        cv=5,
        refit=true,
        n_iter=10,
        random_state=nothing
    )
        new(estimator, param_grid, scoring, cv, refit, n_iter, random_state, -Inf, Vector{Any}(), Dict())
    end
end

function (rs::RandomSearchCV)(X, y)
    all_params = _generate_param_combinations(rs.param_grid)
    n_combinations = min(rs.n_iter, length(all_params))
    
    if !isnothing(rs.random_state)
        Random.seed!(rs.random_state)
    end
    
    selected_params = shuffle!(all_params)[1:n_combinations]
    
    scores = zeros(n_combinations)
    println("Evaluating $n_combinations random combinations.")
    Threads.@threads for i in 1:n_combinations            
        params = selected_params[i]
        model = deepcopy(rs.estimator)
        _set_params!(params)
        
        cv_scores = cross_val_score(model, X, y; cv=rs.cv, scoring=rs.scoring)
        scores[i] = mean(cv_scores)
    end
    
    best_idx = argmax(scores)
    rs.best_score = scores[best_idx]
    rs.best_params = selected_params[best_idx]
    
    rs.cv_results = Dict(
        "mean_test_score" => scores,
        "params" => selected_params
    )
    
    if rs.refit
        _set_params!(rs.best_params)
        rs.estimator(X, y)
    end
    
    return rs
end
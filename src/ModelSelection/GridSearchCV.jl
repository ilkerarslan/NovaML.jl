using Statistics: mean
    using ...Metrics: accuracy_score
    using ...ModelSelection: cross_val_score
    
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
        println("There are $n_combinations combinations.")
        Threads.@threads for i in 1:n_combinations            
            params = all_params[i]
            model = deepcopy(gs.estimator)
            _set_params!(model, params)
            
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
            _set_params!(gs.estimator, gs.best_params)
            gs.estimator(X, y)
        end
        
        return gs
    end
    
    function _generate_param_combinations(param_grid)
        all_params = []
        for grid in param_grid
            component = grid[1]
            param_list = grid[2:end]
            params = _param_product(component, param_list)
            append!(all_params, params)
        end
        return all_params
    end
    
    function _param_product(component, param_list)
        keys = [p[1] for p in param_list]
        values = [p[2] for p in param_list]
        combinations = Iterators.product(values...)
        return [[component, zip(keys, combo)...] for combo in combinations]
    end
    
    function _set_params!(estimator, params)
        component = params[1]
        for (param, value) in params[2:end]
            setproperty!(component, param, value)
        end
    end
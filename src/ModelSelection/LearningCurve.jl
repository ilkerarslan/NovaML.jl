using ...NovaML.ModelSelection: StratifiedKFold
using ...NovaML.Metrics: accuracy_score
using Statistics

function learning_curve(
    estimator,
    X,
    y;
    train_sizes=collect(range(0.1, 1.0, length=5)),
    cv=5,
    scoring=accuracy_score,
    n_jobs=nothing,
    verbose=0,
    shuffle=false,
    random_state=nothing)

    n_samples = size(X, 1)
    
    # Convert train_sizes to absolute numbers if they're fractions
    if eltype(train_sizes) <: AbstractFloat
        train_sizes_abs = round.(Int, train_sizes .* n_samples)
    else
        train_sizes_abs = train_sizes
    end
    train_sizes_abs = unique(train_sizes_abs)
    
    # Create CV iterator
    if cv isa Integer
        cv = StratifiedKFold(n_splits=cv)
    end
    
    # Initialize results
    train_scores = []
    test_scores = []
    
    # Perform CV for each training set size
    for train_size in train_sizes_abs
        train_scores_size = []
        test_scores_size = []
        
        for (train_idx, test_idx) in cv(y)
            if shuffle
                # Implement shuffling logic here if needed
            end
            
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]
            
            # Use only a subset of the training data
            subset_size = min(train_size, length(train_idx))
            subset_idx = train_idx[1:subset_size]
            X_train_subset, y_train_subset = X[subset_idx, :], y[subset_idx]
            
            # Fit the estimator
            estimator(X_train_subset, y_train_subset)
            
            # Calculate scores
            train_score = scoring(estimator(X_train_subset), y_train_subset)
            test_score = scoring(estimator(X_test), y_test)
            
            push!(train_scores_size, train_score)
            push!(test_scores_size, test_score)
        end
        
        push!(train_scores, train_scores_size)
        push!(test_scores, test_scores_size)
    end
    
    return train_sizes_abs, train_scores, test_scores
end
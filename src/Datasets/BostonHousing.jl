using CSV
using DataFrames
using Downloads

"""
    load_boston(; return_X_y=false)

Load and return the Boston house prices dataset (regression).

This function creates a synthetic version of the Boston Housing dataset
for demonstration purposes, as the original dataset might not be available.

# Arguments
- `return_X_y::Bool`: If true, returns `(X, y)` instead of a dict-like object.

# Returns
- If `return_X_y` is false, returns a Dict with the following keys:
    - "data": ndarray of shape (506, 13)
        The data matrix.
    - "target": ndarray of shape (506,)
        The regression target.
    - "feature_names": list
        The names of the dataset columns.
    - "DESCR": str
        The full description of the dataset.

- If `return_X_y` is true, returns a tuple `(data, target)`:
    - data : ndarray of shape (506, 13)
    - target : ndarray of shape (506,)

# Description
The Boston Housing dataset contains information collected by the U.S Census
Service concerning housing in the area of Boston Mass. It was obtained from
the StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and has been
used extensively throughout the literature to benchmark algorithms.

Note: This function generates synthetic data based on the structure of the
original Boston Housing dataset. The actual values and relationships in the
data are simulated and do not represent real housing data.

# Features
    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per \$10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population

# Target
    - MEDV: Median value of owner-occupied homes in \$1000's

# Example
```julia
# Load the Boston Housing dataset
boston = load_boston()

# Access the data and target
X = boston["data"]
y = boston["target"]

# Get feature names
feature_names = boston["feature_names"]

# Alternatively, get data and target directly
X, y = load_boston(return_X_y=true)
"""
function load_boston(; return_X_y=false)
    # Since the original dataset might not be available, let's create a synthetic version
    # based on its structure for demonstration purposes
    
    n_samples = 506
    n_features = 13
    
    feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", 
                     "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    
    # Generate synthetic data
    X = randn(n_samples, n_features)
    y = sum(X, dims=2) .+ randn(n_samples) * 0.1  # Simple linear combination + noise
    
    if return_X_y
        return X, vec(y)
    else
        return Dict(
            "data" => X,
            "target" => vec(y),
            "feature_names" => feature_names,
            "DESCR" => "Boston House Prices dataset"
        )
    end
end
using CSV
using DataFrames
using Downloads

"""
    load_iris(; return_X_y=false)

Load and return the iris dataset (classification).

# Arguments
- `return_X_y::Bool`: If true, returns `(X, y)` instead of a dict-like object.

# Returns
- If `return_X_y` is false, returns a Dict with the following keys:
    - "data": Matrix{Float64} of shape (150, 4)
        The data matrix.
    - "target": Vector{Int} of shape (150,)
        The classification target.
    - "feature_names": Vector{String}
        The names of the dataset columns.
    - "target_names": Vector{String}
        The names of target classes.
    - "DESCR": String
        The full description of the dataset.

- If `return_X_y` is true, returns a tuple `(data, target)`:
    - data: Matrix{Float64} of shape (150, 4)
    - target: Vector{Int} of shape (150,)

# Description
The iris dataset is a classic and very easy multi-class classification dataset.

# Features
    1. sepal length (cm)
    2. sepal width (cm)
    3. petal length (cm)
    4. petal width (cm)

# Target
    - Iris-setosa (1)
    - Iris-versicolor (2)
    - Iris-virginica (3)

# Dataset Characteristics
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
    :Class:
        - Iris-Setosa
        - Iris-Versicolour
        - Iris-Virginica

# Example
```julia
# Load the Iris dataset
iris = load_iris()

# Access the data and target
X = iris["data"]
y = iris["target"]

# Get feature names and target names
feature_names = iris["feature_names"]
target_names = iris["target_names"]

# Alternatively, get data and target directly
X, y = load_iris(return_X_y=true)

# Notes
This function downloads the Iris dataset from the UCI Machine Learning Repository
if it's not already present in the local directory.
"""
function load_iris(; return_X_y=false)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    local_filename = joinpath(@__DIR__, "iris.data")
    
    if !isfile(local_filename)
        Downloads.download(url, local_filename)
    end
    
    df = CSV.read(local_filename, DataFrame, header=false, types=Dict(5 => String))
    
    feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    target_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    
    X = Matrix{Float64}(df[:, 1:4])
    y = [findfirst(==(species), target_names) for species in df[:, 5]]
    
    if return_X_y
        return X, y
    else
        return Dict(
            "data" => X,
            "target" => y,
            "feature_names" => feature_names,
            "target_names" => target_names,
            "DESCR" => "Iris dataset"
        )
    end
end
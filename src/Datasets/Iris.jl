# src/Datasets/Iris.jl

using CSV
using DataFrames
using Downloads

"""
    load_iris(; return_X_y=false)

Load and return the Iris dataset.

The Iris dataset is a classic and very easy multi-class classification dataset.

Parameters:
-----------
return_X_y : bool, default=false
    If True, returns `(X, y)` instead of a dict-like object.

Returns:
--------
data : Dict or Tuple
    Dict with the following items:
        data : Matrix{Float64}
            The feature matrix (150 samples, 4 features).
        target : Vector{Int}
            The classification labels (150 samples).
        feature_names : Vector{String}
            The names of the dataset columns.
        target_names : Vector{String}
            The meaning of the labels.
    If `return_X_y` is True, returns a tuple `(X, y)` instead.

Description:
------------
The dataset contains 3 classes of 50 instances each, where each class refers to a 
type of iris plant. One class is linearly separable from the other 2; the latter 
are NOT linearly separable from each other.

Attribute Information:
    1. sepal length in cm
    2. sepal width in cm
    3. petal length in cm
    4. petal width in cm
    5. class:
        - Iris Setosa
        - Iris Versicolour
        - Iris Virginica
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
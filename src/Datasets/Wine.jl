using CSV
using DataFrames
using Downloads

"""
    load_wine(; return_X_y=false)

Load and return the wine dataset (classification).

# Arguments
- `return_X_y::Bool`: If true, returns `(X, y)` instead of a dict-like object.

# Returns
- If `return_X_y` is false, returns a Dict with the following keys:
    - "data": Matrix{Float64} of shape (178, 13)
        The data matrix.
    - "target": Vector{Int} of shape (178,)
        The classification target.
    - "feature_names": Vector{String}
        The names of the dataset columns.
    - "target_names": Vector{String}
        The names of target classes.
    - "DESCR": String
        The full description of the dataset.

- If `return_X_y` is true, returns a tuple `(data, target)`:
    - data: Matrix{Float64} of shape (178, 13)
    - target: Vector{Int} of shape (178,)

# Description
This dataset is a classic and very easy multi-class classification dataset.

# Features
    1) Alcohol
    2) Malic acid
    3) Ash
    4) Alcalinity of ash
    5) Magnesium
    6) Total phenols
    7) Flavanoids
    8) Nonflavanoid phenols
    9) Proanthocyanins
    10) Color intensity
    11) Hue
    12) OD280/OD315 of diluted wines
    13) Proline

# Target
    - class 1 (0)
    - class 2 (1)
    - class 3 (2)

# Dataset Characteristics
    :Number of Instances: 178
    :Number of Attributes: 13 numeric, predictive attributes and the class
    :Attribute Information:
        - Alcohol
        - Malic acid
        - Ash
        - Alcalinity of ash
        - Magnesium
        - Total phenols
        - Flavanoids
        - Nonflavanoid phenols
        - Proanthocyanins
        - Color intensity
        - Hue
        - OD280/OD315 of diluted wines
        - Proline

    :Class:
        - class 1
        - class 2
        - class 3

# Example
```julia
# Load the Wine dataset
wine = load_wine()

# Access the data and target
X = wine["data"]
y = wine["target"]

# Get feature names and target names
feature_names = wine["feature_names"]
target_names = wine["target_names"]

# Alternatively, get data and target directly
X, y = load_wine(return_X_y=true)

# Notes
This function downloads the Wine dataset from the UCI Machine Learning Repository
if it's not already present in the local directory.

The data set contains the results of a chemical analysis of wines grown
in a specific area of Italy. Three types of wine are represented in the 178
samples, with the results of 13 chemical analyses recorded for each sample.

The classes are ordered and not balanced (class 1 has 59 samples, class 2
has 71 samples, and class 3 has 48 samples).

This dataset is also excellent for visualization techniques.
"""
function load_wine(; return_X_y=false)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    local_filename = joinpath(@__DIR__, "wine.data")
    
    if !isfile(local_filename)
        Downloads.download(url, local_filename)
    end
    
    df = CSV.read(local_filename, DataFrame, header=false)
    
    feature_names = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
        "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"
    ]
    target_names = ["class 1", "class 2", "class 3"]
    
    X = Matrix{Float64}(df[:, 2:end])
    y = Vector{Int}(df[:, 1])
    
    if return_X_y
        return X, y
    else
        return Dict(
            "data" => X,
            "target" => y,
            "feature_names" => feature_names,
            "target_names" => target_names,
            "DESCR" => "Wine dataset"
        )
    end
end
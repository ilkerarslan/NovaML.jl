# src/Datasets/WisconsinBreastCancer.jl
using CSV
using DataFrames
using Downloads

"""
    load_breast_cancer(; return_X_y=false)

Load and return the Wisconsin Breast Cancer dataset (classification).

# Arguments
- `return_X_y::Bool`: If true, returns `(X, y)` instead of a dict-like object.

# Returns
- If `return_X_y` is false, returns a Dict with the following keys:
    - "data": Matrix{Float64} of shape (569, 30)
        The data matrix.
    - "target": Vector{Bool} of shape (569,)
        The classification target.
    - "feature_names": Vector{String}
        The names of the dataset columns.
    - "target_names": Vector{String}
        The names of target classes.
    - "DESCR": String
        The full description of the dataset.

- If `return_X_y` is true, returns a tuple `(data, target)`:
    - data: Matrix{Float64} of shape (569, 30)
    - target: Vector{Bool} of shape (569,)

# Description
The Wisconsin Breast Cancer dataset is a classic and very easy binary classification dataset.

# Features
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 
They describe characteristics of the cell nuclei present in the image.

Ten real-valued features are computed for each cell nucleus:
    1) radius (mean of distances from center to points on the perimeter)
    2) texture (standard deviation of gray-scale values)
    3) perimeter
    4) area
    5) smoothness (local variation in radius lengths)
    6) compactness (perimeter^2 / area - 1.0)
    7) concavity (severity of concave portions of the contour)
    8) concave points (number of concave portions of the contour)
    9) symmetry
    10) fractal dimension ("coastline approximation" - 1)

The mean, standard error, and "worst" or largest (mean of the three largest values) of these
features were computed for each image, resulting in 30 features.

# Target
    - 0: benign
    - 1: malignant

# Dataset Characteristics
    :Number of Instances: 569
    :Number of Attributes: 30 numeric, predictive attributes and the class
    :Attribute Information: 10 real-valued features are computed for each cell nucleus:
        a) radius (mean of distances from center to points on the perimeter)
        b) texture (standard deviation of gray-scale values)
        c) perimeter
        d) area
        e) smoothness (local variation in radius lengths)
        f) compactness (perimeter^2 / area - 1.0)
        g) concavity (severity of concave portions of the contour)
        h) concave points (number of concave portions of the contour)
        i) symmetry
        j) fractal dimension ("coastline approximation" - 1)

    The mean, standard error, and "worst" or largest (mean of the three
    largest values) of these features were computed for each image,
    resulting in 30 features. For instance, field 3 is Mean Radius, field
    13 is Radius SE, field 23 is Worst Radius.

    :Class Distribution: 212 Malignant, 357 Benign

# Example
```julia
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()

# Access the data and target
X = breast_cancer["data"]
y = breast_cancer["target"]

# Get feature names and target names
feature_names = breast_cancer["feature_names"]
target_names = breast_cancer["target_names"]

# Alternatively, get data and target directly
X, y = load_breast_cancer(return_X_y=true)

# Notes

This function downloads the Wisconsin Breast Cancer dataset from the UCI Machine Learning Repository
if it's not already present in the local directory.

The dataset was created by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian
at the University of Wisconsin-Madison.
"""
function load_breast_cancer(; return_X_y=false)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    local_filename = joinpath(@__DIR__, "wdbc.data")
    
    if !isfile(local_filename)
        Downloads.download(url, local_filename)
    end
    
    df = CSV.read(local_filename, DataFrame, header=false)
    
    feature_names = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
        "radius error", "texture error", "perimeter error", "area error",
        "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error",
        "worst radius", "worst texture", "worst perimeter", "worst area",
        "worst smoothness", "worst compactness", "worst concavity",
        "worst concave points", "worst symmetry", "worst fractal dimension"
    ]
    
    X = Matrix(df[:, 3:end])
    y = df[:, 2] .== "M"  # Convert to boolean: true for Malignant, false for Benign
    
    if return_X_y
        return X, y
    else
        return Dict(
            "data" => X,
            "target" => y,
            "feature_names" => feature_names,
            "target_names" => ["benign", "malignant"],
            "DESCR" => "Wisconsin Breast Cancer dataset"
        )
    end
end
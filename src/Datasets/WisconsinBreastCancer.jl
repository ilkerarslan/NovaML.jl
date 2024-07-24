# src/Datasets/WisconsinBreastCancer.jl
using CSV
using DataFrames
using Downloads

"""
    load_breast_cancer(; return_X_y=false)

Load and return the Wisconsin Breast Cancer dataset.

The Wisconsin Breast Cancer dataset is a classic and very easy binary 
classification dataset.

Parameters:
-----------
return_X_y : bool, default=false
    If True, returns `(X, y)` instead of a dict-like object.

Returns:
--------
data : Dict or Tuple
    Dict with the following items:
        data : DataFrame
            The feature matrix (569 samples, 30 features).
        target : Vector
            The classification labels (569 samples).
        feature_names : Vector
            The names of the dataset columns.
        target_names : Vector
            The meaning of the labels.
    If `return_X_y` is True, returns a tuple `(X, y)` instead.

Description:
------------
The dataset contains features computed from a digitized image of a fine 
needle aspirate (FNA) of a breast mass. They describe characteristics of 
the cell nuclei present in the image.

Number of Instances: 569
Number of Attributes: 30 numeric, predictive attributes and the class
Attribute Information:
    - radius (mean of distances from center to points on the perimeter)
    - texture (standard deviation of gray-scale values)
    - perimeter
    - area
    - smoothness (local variation in radius lengths)
    - compactness (perimeter^2 / area - 1.0)
    - concavity (severity of concave portions of the contour)
    - concave points (number of concave portions of the contour)
    - symmetry
    - fractal dimension ("coastline approximation" - 1)

The mean, standard error, and "worst" or largest (mean of the three worst/largest 
values) of these features were computed for each image, resulting in 30 features. 
For instance, field 0 is Mean Radius, field 10 is Radius SE, field 20 is Worst Radius.

Class:
    - WDBC-Malignant
    - WDBC-Benign

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
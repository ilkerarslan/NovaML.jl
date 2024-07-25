using CSV
using DataFrames
using Downloads

"""
    load_wine(; return_X_y=false)

Load and return the Wine dataset.

The Wine dataset is a classic and very easy multi-class classification dataset.

Parameters:
-----------
return_X_y : bool, default=false
    If True, returns `(X, y)` instead of a dict-like object.

Returns:
--------
data : Dict or Tuple
    Dict with the following items:
        data : Matrix{Float64}
            The feature matrix (178 samples, 13 features).
        target : Vector{Int}
            The classification labels (178 samples).
        feature_names : Vector{String}
            The names of the dataset columns.
        target_names : Vector{String}
            The meaning of the labels.
    If `return_X_y` is True, returns a tuple `(X, y)` instead.

Description:
------------
This dataset is the result of a chemical analysis of wines grown in the same region in Italy 
but derived from three different cultivars. The analysis determined the quantities of 13 
constituents found in each of the three types of wines.

Attribute Information:
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

Target classes:
    - class 1
    - class 2
    - class 3
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
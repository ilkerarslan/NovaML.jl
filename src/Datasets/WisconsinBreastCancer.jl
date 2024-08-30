# src/Datasets/WisconsinBreastCancer.jl
using CSV
using DataFrames
using Downloads

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
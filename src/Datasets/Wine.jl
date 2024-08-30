using CSV
using DataFrames
using Downloads

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
using CSV
using DataFrames
using Downloads

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
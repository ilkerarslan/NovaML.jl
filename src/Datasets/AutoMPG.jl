using Downloads

function load_auto_mpg(; return_X_y=false)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

    local_filename = joinpath(@__DIR__, "auto-mpg.data")

    if !isfile(local_filename)
        Downloads.download(url, local_filename)
    end

    data = Vector{Vector{Float64}}()
    target = Vector{Float64}()
    car_names = Vector{String}()

    for line in eachline(local_filename)
        fields = split(line, r"\s+"; limit=9)
        fields = filter(!isempty, fields)

        push!(target, parse(Float64, fields[1]))

        row = Float64[]
        for val in fields[2:8]
            if val == "?"
                push!(row, NaN)
            else
                push!(row, parse(Float64, val))
            end
        end
        push!(data, row)

        push!(car_names, fields[9])
    end

    X = reduce(hcat, data)'
    y = target

    hp_mean = mean(filter(!isnan, X[:, 3]))
    X[isnan.(X[:, 3]), 3] .= hp_mean

    feature_names = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin"
    ]

    description = """
    Auto MPG Dataset
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    The dataset contains fuel consumption information in miles per gallon and various characteristics
    for automobiles between 1970 and 1982.
    
    Number of Instances: $(size(X, 1))
    Number of Features: $(size(X, 2))
    
    Features:
    - cylinders: Number of cylinders
    - displacement: Engine displacement in cubic inches
    - horsepower: Engine horsepower
    - weight: Vehicle weight in pounds
    - acceleration: Time to accelerate from 0 to 60 mph (seconds)
    - model_year: Model year (70 = 1970, etc.)
    - origin: Origin of car (1: American, 2: European, 3: Japanese)
    
    Target:
    - mpg: Fuel efficiency in miles per gallon
    
    Missing Values:
    - horsepower: Contains some missing values (marked with '?'), replaced with mean
    
    The original dataset is available at:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg
    """

    if return_X_y
        return X, y
    else
        return Dict(
            "data" => X,
            "target" => y,
            "feature_names" => feature_names,
            "car_names" => car_names,
            "DESCR" => description
        )
    end
end
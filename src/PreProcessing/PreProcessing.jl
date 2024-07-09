module PreProcessing
    
include("MinMaxScaler.jl")
include("StandardScaler.jl")
include("LabelEncoder.jl")
include("OneHotEncoder.jl")

export MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder

end
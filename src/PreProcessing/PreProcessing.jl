module PreProcessing
    
include("MinMaxScaler.jl")
include("StandardScaler.jl")
include("LabelEncoder.jl")
include("OneHotEncoder.jl")
include("PolynomialFeatures.jl")

export MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures

end
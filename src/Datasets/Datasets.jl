module Datasets

include("BostonHousing.jl")
include("Iris.jl")
include("WisconsinBreastCancer.jl")

export load_boston, load_breast_cancer, load_iris

end
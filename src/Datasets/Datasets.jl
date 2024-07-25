module Datasets

include("BostonHousing.jl")
include("Iris.jl")
include("Wine.jl")
include("WisconsinBreastCancer.jl")

export load_boston, load_breast_cancer, load_iris, load_wine

end
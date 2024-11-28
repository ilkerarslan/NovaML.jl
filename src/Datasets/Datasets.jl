module Datasets

include("AutoMPG.jl")
include("BostonHousing.jl")
include("Iris.jl")
include("MakeBlobs.jl")
include("MakeMoons.jl")
include("Wine.jl")
include("WisconsinBreastCancer.jl")

export load_auto_mpg, load_boston, load_breast_cancer, load_iris, load_wine, make_blobs, make_moons

end
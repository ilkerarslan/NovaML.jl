module Ensemble

include("AdaBoostClassifier.jl")
include("BaggingClassifier.jl")
include("GradientBoostingClassifier.jl")
include("GradientBoostingRegressor.jl")
include("RandomForestClassifier.jl")
include("RandomForestRegressor.jl")
include("VotingClassifier.jl")

export AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, VotingClassifier

end
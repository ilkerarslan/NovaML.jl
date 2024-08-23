module Ensemble

include("AdaBoostClassifier.jl")
include("BaggingClassifier.jl")
include("GradientBoostingClassifier.jl")
include("RandomForestClassifier.jl")
include("RandomForestRegressor.jl")
include("VotingClassifier.jl")

export AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, VotingClassifier

end
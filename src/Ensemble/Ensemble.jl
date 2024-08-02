module Ensemble

include("AdaBoostClassifier.jl")
include("BaggingClassifier.jl")
include("GradientBoostingClassifier.jl")
include("RandomForestClassifier.jl")
include("VotingClassifier.jl")

export AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

end
module Ensemble

include("AdaBoostClassifier.jl")
include("BaggingClassifier.jl")
include("RandomForestClassifier.jl")
include("VotingClassifier.jl")

export AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier

end
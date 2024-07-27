module Ensemble

include("BaggingClassifier.jl")
include("RandomForestClassifier.jl")
include("VotingClassifier.jl")

export BaggingClassifier, RandomForestClassifier, VotingClassifier

end
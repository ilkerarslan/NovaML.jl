module Ensemble

include("RandomForestClassifier.jl")
include("VotingClassifier.jl")

export RandomForestClassifier, VotingClassifier

end
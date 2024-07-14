module ModelSelection

include("CrossValScore.jl")
include("StratifiedKFold.jl")
include("TrainTestSplit.jl")

export cross_val_score
export train_test_split
export stratified_kfold

end # of module ModelSelection
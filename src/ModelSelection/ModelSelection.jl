module ModelSelection

include("TrainTestSplit.jl")
export train_test_split

include("StratifiedKFold.jl")
export stratified_kfold

end # of module ModelSelection
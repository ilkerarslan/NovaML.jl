module ModelSelection

include("StratifiedKFoldModel.jl")
export StratifiedKFold

include("CrossValScore.jl")
include("LearningCurve.jl")
include("TrainTestSplit.jl")
include("ValidationCurve.jl")
export cross_val_score, learning_curve, train_test_split, validation_curve

include("GridSearchCV.jl")
include("RandomSearchCV.jl")
export GridSearchCV, RandomSearchCV

end # of module ModelSelection
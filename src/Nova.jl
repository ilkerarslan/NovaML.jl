module Nova

include("_types.jl")
include("_methods.jl")

export AbstractModel, AbstractMultiClass, AbstractScaler
export linearactivation, sigmoid, net_input, softmax

include("Tree/Tree.jl")
include("Ensemble/Ensemble.jl")
include("LinearModel/LinearModel.jl")
include("MultiClass.jl")
include("Neighbors/Neighbors.jl")


include("Metrics.jl")
include("ModelSelection.jl")
include("PreProcessing.jl")


export LinearModel, ModelSelection, PreProcessing, Metrics, MultiClass
export Tree, Ensemble, KNeighborsClassifier

end

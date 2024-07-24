module NovaML

include("_types.jl")
include("_methods.jl")

export AbstractModel, AbstractMultiClass, AbstractScaler
export linearactivation, sigmoid, net_input, softmax, default_score
export _generate_param_combinations, _param_product, _set_params!

include("Tree/Tree.jl")
include("Ensemble/Ensemble.jl")
include("LinearModel/LinearModel.jl")
include("MultiClass/MultiClass.jl")
include("Neighbors/Neighbors.jl")
include("SVM/SVM.jl")

export Tree, Ensemble, LinearModel, MultiClass, Neighbors, SVM

include("Impute.jl")
include("Metrics/Metrics.jl")
include("ModelSelection/ModelSelection.jl")
include("PreProcessing/PreProcessing.jl")

export Impute, Metrics, ModelSelection, PreProcessing

include("Decomposition/Decomposition.jl")
export Decomposition

include("PipeLines/PipeLines.jl")
export Pipelines

include("Utils/Utils.jl")
export Utils

end # of module NovaML

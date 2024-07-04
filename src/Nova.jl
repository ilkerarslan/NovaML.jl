module Nova

include("LinearModel.jl")
include("ModelSelection.jl")
include("PreProcessing.jl")
include("Metrics.jl")

export LinearModel, ModelSelection, PreProcessing, Metrics

end

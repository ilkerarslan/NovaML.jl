using Documenter, NovaML

makedocs(
    sitename = "NovaML.jl",
    format = Documenter.HTML(),
    modules = [NovaML],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => [
            "Linear Models" => "api/linear_models.md",
            "Tree Models" => "api/tree_models.md",
            "Ensemble Models" => "api/ensemble_models.md",
            "SVM" => "api/svm.md",
            "Neighbors" => "api/neighbors.md",
            "Preprocessing" => "api/preprocessing.md",
            "Model Selection" => "api/model_selection.md",
            "Metrics" => "api/metrics.md",
        ],
        "Examples" => [
            "Classification" => "examples/classification.md",
            "Regression" => "examples/regression.md",
            "Clustering" => "examples/clustering.md",
        ],
        "Advanced Topics" => [
            "Custom Models" => "advanced/custom_models.md",
            "Pipelines" => "advanced/pipelines.md",
        ],
    ],
    build = "build",
    source = "src"
)

deploydocs(
    repo = "github.com/ilkerarslan/NovaML.jl.git",
    devbranch = "master"
)
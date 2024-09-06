using Documenter, NovaML

makedocs(
    sitename = "NovaML.jl",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [NovaML],
    pages = [
        "Home" => "index.md",
        "User Guide" => [
            "Getting Started" => "user_guide/getting_started.md",
            "Core Concepts" => "user_guide/core_concepts.md",
            "Data Preprocessing" => "user_guide/preprocessing.md",
            "Model Training" => "user_guide/model_training.md",
            "Model Evaluation" => "user_guide/model_evaluation.md",
            "Feature Engineering" => "user_guide/feature_engineering.md",
            "Pipelines" => "user_guide/pipelines.md",
            "Hyperparameter Tuning" => "user_guide/hyperparameter_tuning.md",
            "Cross-Validation" => "user_guide/cross_validation.md",
        ],
        "Models" => [
            "Linear Models" => "models/linear_models.md",
            "Tree-Based Models" => "models/tree_models.md",
            "Ensemble Methods" => "models/ensemble_methods.md",
            "Support Vector Machines" => "models/svm.md",
            "Clustering" => "models/clustering.md",
            "Dimensionality Reduction" => "models/dimensionality_reduction.md",
        ],
        "API Reference" => "api.md",
        "Contributing" => "contribute.md",
    ],
)

deploydocs(
    repo = "github.com/ilkerarslan/NovaML.jl.git",
    devbranch = "master",
)
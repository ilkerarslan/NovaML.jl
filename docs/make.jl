using Documenter, NovaML
using NovaML.Cluster
using NovaML.Decomposition
using NovaML.Ensemble
using NovaML.Tree

makedocs(
    sitename = "NovaML.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://ilkerarslan.github.io/NovaML.jl/stable/",
        repolink = "https://github.com/ilkerarslan/NovaML.jl",
        assets = String[],
        analytics = "UA-XXXXXXXXX-X",
    ),
    modules = [NovaML],
    authors = "Ilker Arslan and contributors",
    repo = "https://github.com/ilkerarslan/NovaML.jl/blob/{commit}{path}#{line}",
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
        "FAQ" => "faq.md",
    ],
    doctest = true,
    linkcheck = true,
)

deploydocs(
    repo = "github.com/ilkerarslan/NovaML.jl.git",
    devbranch = "master",
    push_preview = true,
)
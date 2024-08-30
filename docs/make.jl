using Documenter, NovaML

makedocs(
    sitename = "NovaML.jl",
    Documenter.HTML(),
    modules = [NovaML],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Overview" => "user_guide/overview.md"
        ],
        "API Reference" => "api_reference.md",
        "Examples" => "examples.md",
        "Contributing" => "contributing.md"
    ]
)

deploydocs(
    repo = "github.com/ilkerarslan/NovaML.jl.git",
)
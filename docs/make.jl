using Documenter, NovaML

makedocs(
    sitename = "NovaML.jl",
    modules = [NovaML],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api_reference.md",
        "Examples" => "examples.md",
    ]
)

deploydocs(
    repo = "github.com/ilkerarslan/NovaML.jl.git",
)
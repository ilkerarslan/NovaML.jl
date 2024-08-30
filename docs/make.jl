using Documenter, NovaML

makedocs(
    sitename = "NovaML.jl",
    modules = [NovaML],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/ilkerarslan/NovaML.jl.git",
)
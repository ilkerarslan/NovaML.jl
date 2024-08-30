using Documenter, NovaML

makedocs(
    sitename = "NovaML.jl",
    format = Documenter.HTML(),
    modules = [NovaML],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md"
    ],
    strict = false 
)

deploydocs(
    repo = "github.com/ilkerarslan/NovaML.jl.git",
)
using UNet
using Documenter

DocMeta.setdocmeta!(UNet, :DocTestSetup, :(using UNet); recursive=true)

makedocs(;
    modules=[UNet],
    authors="Arnold",
    repo="https://github.com/a-r-n-o-l-d/UNet.jl/blob/{commit}{path}#{line}",
    sitename="UNet.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://a-r-n-o-l-d.github.io/UNet.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/a-r-n-o-l-d/UNet.jl",
)

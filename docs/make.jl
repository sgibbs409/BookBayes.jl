using Documenter
using BookBayes


if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

#push!(LOAD_PATH, "../src/")
makedocs(
    modules = [BookBayes],
    sitename = "BookBayesjl Documentation",
    pages = [
        "Index" => "index.md",
        "Another page" => "anotherPage.md",
    ],

    format = Documenter.HTML(
        #prettyurls = true
        canonical = "https://sgibbs409.github.io/BookBayes.jl/stable/",
        prettyurls = !("local" in ARGS)
    ),

    strict = !("strict=false" in ARGS),
    doctest = ("doctest=only" in ARGS) ? :only : true,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/sgibbs409/BookBayes.jl.git",
    devbranch="main",
    target="build",
    push_preview = true,
    #versions = ["stable" => "v^", "v#.#", "dev" => "v^"],
    #versions = ["v#.#", "dev" => "v^"],
    #versions = ["stable" => "v^", "v#.#"]
)

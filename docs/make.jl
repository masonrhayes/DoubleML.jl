using Documenter
using DoubleML
using Documenter.Remotes: GitHub
using PlutoStaticHTML

# Build Pluto notebooks before generating documentation
println("Building Pluto notebooks...")

# Define notebook paths and their destination names
notebooks = [
    ("PLR Introduction", joinpath(@__DIR__, "..", "examples", "PLR", "plr_introduction.jl")),
    ("LPLR Introduction", joinpath(@__DIR__, "..", "examples", "LPLR", "lplr_introduction.jl")),
    ("IRM Introduction", joinpath(@__DIR__, "..", "examples", "IRM", "irm_introduction.jl")),
]

# Output directory for notebooks in documentation
notebooks_output_dir = joinpath(@__DIR__, "src", "notebooks")
mkpath(notebooks_output_dir)

# Build each notebook and copy to docs/src/notebooks/
for (title, notebook_path) in notebooks
    if isfile(notebook_path)
        println("  Building: $title")

        # Get the directory containing the notebook
        notebook_dir = dirname(notebook_path)

        # Build options - output goes next to the notebook
        bopts = PlutoStaticHTML.BuildOptions(
            notebook_dir;
            output_format = PlutoStaticHTML.documenter_output,
            add_documenter_css = true,
        )

        # Build the notebook
        PlutoStaticHTML.build_notebooks(bopts, [notebook_path])

        # The generated file has the same name but with .md extension
        notebook_name = basename(notebook_path)
        generated_md = replace(notebook_name, ".jl" => ".md")
        generated_path = joinpath(notebook_dir, generated_md)

        # Copy to docs/src/notebooks/
        dest_path = joinpath(notebooks_output_dir, generated_md)
        cp(generated_path, dest_path; force = true)
        println("    ✓ Copied to: $dest_path")
    else
        @warn "Notebook not found: $notebook_path"
    end
end

println("✓ Pluto notebooks built successfully")

DocMeta.setdocmeta!(DoubleML, :DocTestSetup, :(using DoubleML, MLJ, DataFrames); recursive = true)

makedocs(
    sitename = "DoubleML.jl",
    modules = [DoubleML],
    authors = "Mason R. Hayes and contributors",
    repo = GitHub("masonrhayes", "DoubleML.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://masonrhayes.github.io/DoubleML.jl",
        edit_link = "main",
        assets = String[],
        ansicolor = true,
        inventory_version = "0.1.0",
    ),
    pages = [
        "Home" => "index.md",
        "User Guide" => "user-guide.md",
        "Tutorials" => [
            "PLR" => "tutorials/plr.md",
            "IRM" => "tutorials/irm.md",
            "LPLR" => "tutorials/lplr.md",
        ],
        "Examples" => [
            "Overview" => "examples.md",
            "PLR Introduction" => "notebooks/plr_introduction.md",
            "LPLR Introduction" => "notebooks/lplr_introduction.md",
            "IRM Introduction" => "notebooks/irm_introduction.md",
        ],
        "API Reference" => "api.md",
    ],
    doctest = true,
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/masonrhayes/DoubleML.jl.git",
    devbranch = "main",
    push_preview = true,
)

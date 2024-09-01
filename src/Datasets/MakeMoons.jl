using Random

"""
    make_moons(;
        n_samples::Union{Int, Tuple{Int, Int}}=100,
        shuffle::Bool=true,
        noise::Union{Float64, Nothing}=nothing,
        random_state::Union{Int, Nothing}=nothing
    )

Generate two interleaving half circles for binary classification.

# Arguments
- `n_samples::Union{Int, Tuple{Int, Int}}`: The total number of points generated or a tuple containing the number of points in each of the two moons.
- `shuffle::Bool`: Whether to shuffle the samples.
- `noise::Union{Float64, Nothing}`: Standard deviation of Gaussian noise added to the data.
- `random_state::Union{Int, Nothing}`: Determines random number generation for dataset creation.

# Returns
- `X::Matrix{Float64}`: The generated samples, of shape (n_samples, 2).
- `y::Vector{Int}`: The integer labels (0 or 1) for class membership of each sample.

# Description
This function generates a binary classification dataset in the shape of two interleaving half moons.
It can be used for testing classification algorithms or as a simple dataset for demonstration purposes.

# Example
```julia
# Generate a simple moon dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Generate a moon dataset with different number of samples in each moon
X, y = make_moons(n_samples=(60, 40), noise=0.1, shuffle=false)

# Notes
- If n_samples is an integer, it generates approximately equal numbers of samples in each moon.
- If the number is odd, the extra sample is added to the first moon.
- If n_samples is a tuple of two integers, it specifies the number of samples for each moon respectively.
- The two moons are generated on a 2D plane. The first moon is a half circle of radius 1 centered at (0, 0),
while the second moon is a half circle of radius 1 centered at (1, 0.5).
- If noise is specified, Gaussian noise with standard deviation noise is added to the data.
"""
function make_moons(; n_samples::Union{Int, Tuple{Int, Int}}=100,
    shuffle::Bool=true,
    noise::Union{Float64, Nothing}=nothing,
    random_state::Union{Int, Nothing}=nothing)
    if random_state !== nothing
        Random.seed!(random_state)
    end
    
    if n_samples isa Tuple
        n_samples_out, n_samples_in = n_samples
    else
        n_samples_out = n_samples_in = n_samples ÷ 2
        n_samples_out += n_samples % 2  # Add extra sample to first moon if odd
    end
    
    # Outer circle
    linspace_out = range(0, π, length=n_samples_out)
    outer_circ_x = cos.(linspace_out)
    outer_circ_y = sin.(linspace_out)
    X_out = hcat(outer_circ_x, outer_circ_y)
    
    # Inner circle
    linspace_in = range(0, π, length=n_samples_in)
    inner_circ_x = 1 .- cos.(linspace_in)
    inner_circ_y = 1 .- sin.(linspace_in) .- 0.5
    X_in = hcat(inner_circ_x, inner_circ_y)
    
    X = vcat(X_out, X_in)
    y = vcat(zeros(Int, n_samples_out), ones(Int, n_samples_in))
    
    if noise !== nothing
        X .+= randn(size(X)) .* noise
    end
    
    if shuffle
        indices = Random.shuffle(1:size(X, 1))
        X = X[indices, :]
        y = y[indices]
    end
    
    return X, y
end
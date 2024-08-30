using Random

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
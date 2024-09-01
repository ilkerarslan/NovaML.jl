module Pipelines

export pipe

mutable struct pipe
    steps::Vector{Any}
end

function pipe(steps...)
    return pipe(collect(steps))
end

function (p::pipe)(X, y=nothing; kwargs...)
    data = X
    for (i, step) in enumerate(p.steps)
        if i == length(p.steps)
            # Last step (usually the model)
            if y !== nothing
                # Training
                step(data, y)
            else
                # Prediction or transformation
                return step(data; kwargs...)
            end
        else
            # Transformation step
            data = step(data)
        end
    end
    return data
end

function Base.last(p::pipe)
    return p.steps[end]
end

# Pretty printing
function Base.show(io::IO, p::pipe)
    println(io, "pipe(")
    for (i, step) in enumerate(p.steps)
        print(io, "  ")
        show(io, step)
        if i < length(p.steps)
            println(io, ",")
        else
            println(io)
        end
    end
    print(io, ")")
end

end # module Pipelines
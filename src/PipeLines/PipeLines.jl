module Pipelines

export pipe

mutable struct pipe
    steps::Vector{Any}
end

function pipe(steps...)
    return pipe(collect(steps))
end

function (p::pipe)(X, y=nothing)
    data = X
    for (i, step) in enumerate(p.steps)
        if i == length(p.steps) && y !== nothing
            # Last step with y available (usually the model)
            step(data, y)
        else
            # Transformation step or prediction
            data = step(data)
        end
    end
    return data
end

# Helper method to get the last step (usually the model)
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
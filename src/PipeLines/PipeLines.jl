module Pipelines

export Pipe

mutable struct Pipe
    steps::Vector{Any}
end

function Pipe(steps...)
    return Pipe(collect(steps))
end

function (pipe::Pipe)(X, y=nothing)
    data = X
    for (i, step) in enumerate(pipe.steps)
        if i == length(pipe.steps) && y !== nothing
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
function Base.last(pipe::Pipe)
    return pipe.steps[end]
end

# Pretty printing
function Base.show(io::IO, pipe::Pipe)
    println(io, "Pipe(")
    for (i, step) in enumerate(pipe.steps)
        print(io, "  ")
        show(io, step)
        if i < length(pipe.steps)
            println(io, ",")
        else
            println(io)
        end
    end
    print(io, ")")
end

end # module Pipelines
module MultiClass

using ..LinearModel 
import ..Nova: sigmoid, net_input


export OneVsRestClassifier

mutable struct OneVsRestClassifier
    estimator::LogisticRegression
    classes::Vector{Any}
    classifiers::Vector{LogisticRegression}
    fitted::Bool
end

function OneVsRestClassifier(; η=0.01, num_iter=100, random_state=nothing, optim_alg=:Batch, batch_size=32, λ=0.01)
    base_estimator = LogisticRegression(η=η, num_iter=num_iter, random_state=random_state, 
                                        optim_alg=optim_alg, batch_size=batch_size, λ=λ)

    return OneVsRestClassifier(base_estimator, [], [], false)    
end

function (m::OneVsRestClassifier)(X::AbstractMatrix, y::AbstractVector)
    m.classes = sort(unique(y))
    m.classifiers = [deepcopy(m.estimator) for _ in 1:length(m.classes)]

    for (i, class) in enumerate(m.classes)
        y_binary = float.(y .== class)
        m.classifiers[i](X, y_binary)
    end

    m.fitted = true
end

function (m::OneVsRestClassifier)(X::AbstractMatrix; type=nothing)
    if !m.fitted
        error("Model is not fitted yet.")
    end

    n_samples = size(X, 1)
    n_classes = length(m.classes)
    scores = zeros(n_samples, n_classes)

    for (i, classifier) in enumerate(m.classifiers)
        scores[:, i] = sigmoid.(net_input(classifier, X))
    end

    if type ≡ nothing 
        _, predicted_indices = findmax(scores, dims=2)
        return [m.classes[idx[2]] for idx in predicted_indices] |> vec
    elseif type == :probs 
        return scores
    end
end


end # of module MultiClass
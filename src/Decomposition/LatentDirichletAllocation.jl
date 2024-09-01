using Random
using LinearAlgebra
using SparseArrays
using SpecialFunctions: digamma
using Distributions: Gamma

"""
    LatentDirichletAllocation

Latent Dirichlet Allocation (LDA) with online variational Bayes algorithm.

LDA is a generative probabilistic model for collections of discrete dataset such as text corpora.
It is also a topic model that is used for discovering abstract topics from a collection of documents.

# Fields
- `n_components::Int`: Number of topics.
- `doc_topic_prior::Union{Float64, Nothing}`: Prior of document topic distribution.
- `topic_word_prior::Union{Float64, Nothing}`: Prior of topic word distribution.
- `learning_method::Symbol`: Method used to update the model: :batch for batch learning, :online for online learning.
- `learning_decay::Float64`: It is a parameter that control the rate at which the learning rate decreases.
- `learning_offset::Float64`: A (positive) parameter that downweights early iterations in online learning.
- `max_iter::Int`: The maximum number of iterations.
- `batch_size::Int`: Number of documents to use in each EM iteration in online learning method.
- `evaluate_every::Int`: How often to evaluate perplexity.
- `total_samples::Float64`: Total number of documents.
- `perp_tol::Float64`: Perplexity tolerance in batch learning.
- `mean_change_tol::Float64`: Stopping tolerance for updating document topic distribution in E-step.
- `max_doc_update_iter::Int`: Max number of iterations for updating document topic distribution in E-step.
- `n_jobs::Union{Int, Nothing}`: The number of jobs to use in the E-step.
- `verbose::Int`: Verbosity level.
- `random_state::Union{Int, Nothing}`: Seed for random number generation.

# Learned attributes
- `components_::Union{Matrix{Float64}, Nothing}`: Topic word distribution. shape = (n_components, n_features)
- `exp_dirichlet_component_::Union{Matrix{Float64}, Nothing}`: Exponential value of expectation of log topic word distribution. shape = (n_components, n_features)
- `n_batch_iter_::Int`: Number of iterations of the EM step.
- `n_iter_::Int`: Number of passes over the dataset.
- `bound_::Float64`: Final perplexity score on training set.
- `n_features_in_::Int`: Number of features seen during fit.
- `feature_names_in_::Union{Vector{String}, Nothing}`: Names of features seen during fit.

# Example
```julia
using NovaML

# Create an LDA model
lda = LatentDirichletAllocation(n_components=10, random_state=42)

# Fit the model to data
doc_topic_distr = lda(X)

# Transform new data
new_doc_topic_distr = lda(new_X)
"""
mutable struct LatentDirichletAllocation
    n_components::Int
    doc_topic_prior::Union{Float64, Nothing}
    topic_word_prior::Union{Float64, Nothing}
    learning_method::Symbol
    learning_decay::Float64
    learning_offset::Float64
    max_iter::Int
    batch_size::Int
    evaluate_every::Int
    total_samples::Float64
    perp_tol::Float64
    mean_change_tol::Float64
    max_doc_update_iter::Int
    n_jobs::Union{Int, Nothing}
    verbose::Int
    random_state::Union{Int, Nothing}
    
    # Learned attributes
    components_::Union{Matrix{Float64}, Nothing}
    exp_dirichlet_component_::Union{Matrix{Float64}, Nothing}
    n_batch_iter_::Int
    n_iter_::Int
    bound_::Float64
    n_features_in_::Int
    feature_names_in_::Union{Vector{String}, Nothing}
    
    # Internal state
    rng::Random.AbstractRNG
    fitted::Bool

    function LatentDirichletAllocation(;
        n_components::Int = 10,
        doc_topic_prior::Union{Float64, Nothing} = nothing,
        topic_word_prior::Union{Float64, Nothing} = nothing,
        learning_method::Symbol = :batch,
        learning_decay::Float64 = 0.7,
        learning_offset::Float64 = 10.0,
        max_iter::Int = 10,
        batch_size::Int = 128,
        evaluate_every::Int = -1,
        total_samples::Float64 = 1e6,
        perp_tol::Float64 = 1e-1,
        mean_change_tol::Float64 = 1e-3,
        max_doc_update_iter::Int = 100,
        n_jobs::Union{Int, Nothing} = nothing,
        verbose::Int = 0,
        random_state::Union{Int, Nothing} = nothing
    )
        @assert n_components > 0 "n_components must be positive"
        @assert learning_method in [:batch, :online] "learning_method must be :batch or :online"
        @assert 0.5 < learning_decay <= 1.0 "learning_decay must be in (0.5, 1.0]"
        @assert learning_offset > 1.0 "learning_offset must be greater than 1.0"
        @assert max_iter > 0 "max_iter must be positive"
        @assert batch_size > 0 "batch_size must be positive"
        
        rng = Random.MersenneTwister(random_state === nothing ? Random.make_seed() : random_state)
        
        new(
            n_components, doc_topic_prior, topic_word_prior, learning_method,
            learning_decay, learning_offset, max_iter, batch_size, evaluate_every,
            total_samples, perp_tol, mean_change_tol, max_doc_update_iter,
            n_jobs, verbose, random_state,
            nothing, nothing, 0, 0, 0.0, 0, nothing,
            rng, false
        )
    end
end

function (lda::LatentDirichletAllocation)(X::AbstractMatrix{T}; type=nothing) where T <: Real
    if !lda.fitted
        # Fit and transform
        return _fit_transform(lda, X)
    else
        # Transform only
        return _transform(lda, X)
    end
end

function _fit_transform(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
    n_samples, n_features = size(X)
    
    if lda.doc_topic_prior === nothing
        lda.doc_topic_prior = 1.0 / lda.n_components
    end
    
    if lda.topic_word_prior === nothing
        lda.topic_word_prior = 1.0 / lda.n_components
    end
    
    lda.n_features_in_ = n_features
    lda.components_ = rand(lda.rng, Gamma(100.0, 0.01), lda.n_components, n_features)
    lda.exp_dirichlet_component_ = exp.(digamma.(lda.components_))
    
    doc_topic_distr = if lda.learning_method == :batch
        _fit_batch(lda, X)
    else
        _fit_online(lda, X)
    end
    
    lda.fitted = true
    return doc_topic_distr
end

"""
    _fit_batch(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
Fit the model to X using batch variational Bayes method.

# Arguments
- `lda::LatentDirichletAllocation`: The LDA model.
- `X::AbstractMatrix{T}`: Document-term matrix.

# Returns
- `Matrix{Float64}`: Document-topic distribution.
"""
function _fit_batch(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
    n_samples, n_features = size(X)
    doc_topic_distr = nothing
    
    for iter in 1:lda.max_iter
        last_bound = lda.bound_
        doc_topic_distr = _e_step(lda, X)
        _m_step(lda, X, doc_topic_distr)
        
        if lda.evaluate_every > 0 && iter % lda.evaluate_every == 0
            lda.bound_ = _perplexity(lda, X, doc_topic_distr)
            lda.n_iter_ = iter
            
            if lda.verbose > 0
                println("Iteration: $(iter), Perplexity: $(exp(-lda.bound_ / sum(X)))")
            end
            
            if last_bound > 0 && abs(last_bound - lda.bound_) / last_bound < lda.perp_tol
                break
            end
        end
    end
    
    return doc_topic_distr
end

"""
    _fit_online(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
Fit the model to X using online variational Bayes method.

# Arguments
- `lda::LatentDirichletAllocation`: The LDA model.
- `X::AbstractMatrix{T}`: Document-term matrix.

# Returns
- `Matrix{Float64}`: Document-topic distribution.
"""
function _fit_online(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
    n_samples, n_features = size(X)
    
    for iter in 1:lda.max_iter
        for idx_slice in Iterators.partition(Random.shuffle(lda.rng, 1:n_samples), lda.batch_size)
            X_slice = X[idx_slice, :]
            doc_topic_distr = _e_step(lda, X_slice)
            
            if iter == 1 && idx_slice[1] == 1
                lda.components_ *= n_samples
            end
            
            rho = (lda.learning_offset + lda.n_batch_iter_)^(-lda.learning_decay)
            lda.components_ *= 1 - rho
            _m_step(lda, X_slice, doc_topic_distr, rho * n_samples / length(idx_slice))
            lda.n_batch_iter_ += 1
        end
        
        if lda.evaluate_every > 0 && iter % lda.evaluate_every == 0
            doc_topic_distr = _e_step(lda, X)
            lda.bound_ = _perplexity(lda, X, doc_topic_distr)
            lda.n_iter_ = iter
            
            if lda.verbose > 0
                println("Iteration: $(iter), Perplexity: $(exp(-lda.bound_ / sum(X)))")
            end
        end
    end
    
    return _e_step(lda, X)
end

"""
    _e_step(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
E-step in EM update.

# Arguments
- `lda::LatentDirichletAllocation`: The LDA model.
- `X::AbstractMatrix{T}`: Document-term matrix.

# Returns
- `Matrix{Float64}`: Document-topic distribution.
"""
function _e_step(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
    n_samples, n_features = size(X)
    doc_topic_distr = ones(n_samples, lda.n_components) / lda.n_components
    
    for _ in 1:lda.max_doc_update_iter
        last_doc_topic_distr = copy(doc_topic_distr)
        for i in 1:n_samples
            doc = X[i, :]
            if nnz(doc) == 0
                continue
            end
            
            topic_distr = doc_topic_distr[i, :]
            
            if doc isa SparseVector
                nz_indices = findnz(doc)[1]
                nz_values = nonzeros(doc)
                topic_word_distr = lda.exp_dirichlet_component_[:, nz_indices]
                topic_distr .*= sum(topic_word_distr .* nz_values', dims=2)
            else
                topic_word_distr = lda.exp_dirichlet_component_
                topic_distr .*= sum(topic_word_distr .* doc', dims=2)
            end
            
            topic_distr ./= sum(topic_distr)
            doc_topic_distr[i, :] = topic_distr
        end
        
        if maximum(abs.(doc_topic_distr - last_doc_topic_distr)) < lda.mean_change_tol
            break
        end
    end
    
    return doc_topic_distr
end

"""
    _m_step(lda::LatentDirichletAllocation, X::AbstractMatrix{T}, doc_topic_distr::Matrix{Float64}, scale::Float64=1.0) where T <: Real
M-step in EM update.

# Arguments
- `lda::LatentDirichletAllocation`: The LDA model.
- `X::AbstractMatrix{T}`: Document-term matrix.
- `doc_topic_distr::Matrix{Float64}`: Document-topic distribution.
- `scale`::Float64: Scaling factor for online update.
"""
function _m_step(lda::LatentDirichletAllocation, X::AbstractMatrix{T}, doc_topic_distr::Matrix{Float64}, scale::Float64=1.0) where T <: Real
    n_samples, n_features = size(X)
    
    if X isa SparseMatrixCSC
        topic_word = doc_topic_distr' * X
    else
        topic_word = doc_topic_distr' * X
    end
    
    topic_word .*= scale
    lda.components_ += topic_word
    
    lda.exp_dirichlet_component_ = exp.(digamma.(lda.components_))
    lda.exp_dirichlet_component_ ./= sum(lda.exp_dirichlet_component_, dims=2)
end

"""
    _transform(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
Transform X to document-topic distribution.

# Arguments
- `lda::LatentDirichletAllocation`: The LDA model.
- `X::AbstractMatrix{T}`: Document-term matrix.

# Returns
- `Matrix{Float64}`: Document-topic distribution.
"""
function _transform(lda::LatentDirichletAllocation, X::AbstractMatrix{T}) where T <: Real
    return _e_step(lda, X)
end

"""
    _perplexity(lda::LatentDirichletAllocation, X::AbstractMatrix{T}, doc_topic_distr::Matrix{Float64}) where T <: Real
Calculate approximate perplexity for data X.

# Arguments
- `lda::LatentDirichletAllocation`: The LDA model.
- `X::AbstractMatrix{T}`: Document-term matrix.
- `doc_topic_distr::Matrix{Float64}`: Document-topic distribution.

# Returns
- `Float64`: The calculated bound.
"""
function _perplexity(lda::LatentDirichletAllocation, X::AbstractMatrix{T}, doc_topic_distr::Matrix{Float64}) where T <: Real
    n_samples, n_features = size(X)
    bound = 0.0
    
    for i in 1:n_samples
        doc = X[i, :]
        if nnz(doc) == 0
            continue
        end
        
        theta = doc_topic_distr[i, :]
        
        if doc isa SparseVector
            nz_indices = findnz(doc)[1]
            nz_values = nonzeros(doc)
            topic_word_distr = lda.exp_dirichlet_component_[:, nz_indices]
            bound += sum(nz_values .* log.(theta' * topic_word_distr))
        else
            topic_word_distr = lda.exp_dirichlet_component_
            bound += sum(doc .* log.(theta' * topic_word_distr))
        end
        
        bound += sum((lda.doc_topic_prior - theta) .* digamma.(theta))
        bound += lgamma(sum(theta)) - sum(lgamma.(theta))
        
        if doc isa SparseVector
            nz_indices = findnz(doc)[1]
            bound += sum((lda.topic_word_prior - lda.components_[:, nz_indices]) .* digamma.(lda.components_[:, nz_indices]))
            bound += sum(lgamma.(sum(lda.components_[:, nz_indices], dims=2)) .- sum(lgamma.(lda.components_[:, nz_indices]), dims=2))
        else
            bound += sum((lda.topic_word_prior - lda.components_) .* digamma.(lda.components_))
            bound += sum(lgamma.(sum(lda.components_, dims=2)) .- sum(lgamma.(lda.components_), dims=2))
        end
    end
    
    return bound
end

"""
    Base.show(io::IO, lda::LatentDirichletAllocation)
Custom show method for LatentDirichletAllocation.

# Arguments
- `io::IO`: The I/O stream
- `lda::LatentDirichletAllocation`: The LDA model to display
"""
function Base.show(io::IO, lda::LatentDirichletAllocation)
    print(io, "LatentDirichletAllocation(n_components=$(lda.n_components), ")
    print(io, "learning_method=$(lda.learning_method), ")
    print(io, "max_iter=$(lda.max_iter), ")
    print(io, "fitted=$(lda.fitted))")
end
using SparseArrays
using LinearAlgebra

mutable struct TfidfVectorizer
    cv::CountVectorizer
    use_idf::Bool
    norm::Union{Nothing, String}
    smooth_idf::Bool
    sublinear_tf::Bool
    idf_::Vector{Float64}
    fitted::Bool

    function TfidfVectorizer(;
        input::String="content",
        encoding::String="utf-8",
        decode_error::String="strict",
        strip_accents::Union{Nothing, String, Function}=nothing,
        lowercase::Bool=true,
        preprocessor::Union{Nothing, Function}=nothing,
        tokenizer::Union{Nothing, Function}=nothing,
        analyzer::Union{String, Function}="word",
        stop_words::Union{Nothing, String, Vector{String}}=nothing,
        token_pattern::String="\\b\\w+\\b",
        ngram_range::Tuple{Int, Int}=(1, 1),
        max_df::Union{Float64, Int}=1.0,
        min_df::Union{Float64, Int}=1,
        max_features::Union{Nothing, Int}=nothing,
        vocabulary::Union{Nothing, Dict{String, Int}}=nothing,
        binary::Bool=false,
        dtype::DataType=Float64,
        use_idf::Bool=true,
        norm::Union{Nothing, String}="l2",
        smooth_idf::Bool=true,
        sublinear_tf::Bool=false
    )
        cv = CountVectorizer(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, analyzer=analyzer,
            max_df=max_df, min_df=min_df, max_features=max_features,
            binary=binary, dtype=dtype
        )
        new(cv, use_idf, norm, smooth_idf, sublinear_tf, Float64[], false)
    end
end

function (tfidf::TfidfVectorizer)(raw_documents::AbstractVector{T} where T <: AbstractString)
    if !tfidf.fitted
        X = tfidf.cv(raw_documents)
        tfidf.idf_ = _calculate_idf(tfidf, X)
        tfidf.fitted = true
        return _transform(tfidf, X)
    else
        # Handle transformation of new documents
        return _transform_new_documents(tfidf, raw_documents)
    end
end

function (tfidf::TfidfVectorizer)(X::AbstractMatrix; type::Symbol=:transform)
    if type == :transform
        if !tfidf.fitted
            throw(ErrorException("TfidfVectorizer is not fitted"))
        end
        return _transform(tfidf, X)
    elseif type == :inverse_transform
        return tfidf.cv(X, type=:inverse_transform)
    else
        throw(ArgumentError("Invalid type. Use :transform or :inverse_transform."))
    end
end

function _calculate_idf(tfidf::TfidfVectorizer, X::AbstractMatrix)
    n_samples, n_features = size(X)
    df = vec(sum(X .> 0, dims=1))
    
    idf = if tfidf.smooth_idf
        log.((n_samples + 1) ./ (df .+ 1)) .+ 1
    else
        log.(n_samples ./ df) .+ 1
    end
    
    return idf
end

function _transform_new_documents(tfidf::TfidfVectorizer, raw_documents::AbstractVector{T} where T <: AbstractString)
    analyzer = _build_analyzer(tfidf.cv)
    n_samples = length(raw_documents)
    n_features = length(tfidf.cv.vocabulary_)
    
    rows = Int[]
    cols = Int[]
    values = Float64[]

    for (doc_idx, doc) in enumerate(raw_documents)
        feature_counter = Dict{Int, Int}()
        doc_terms = analyzer(doc)
        for term in doc_terms
            feature_idx = get(tfidf.cv.vocabulary_, term, 0)
            if feature_idx != 0
                feature_counter[feature_idx] = get(feature_counter, feature_idx, 0) + 1
            end
        end
        
        for (feature_idx, count) in feature_counter
            push!(rows, doc_idx)
            push!(cols, feature_idx)
            tf = tfidf.cv.binary ? 1.0 : (tfidf.sublinear_tf ? 1.0 + log(count) : Float64(count))
            idf = tfidf.use_idf ? tfidf.idf_[feature_idx] : 1.0
            push!(values, tf * idf)
        end
    end
    
    X = sparse(rows, cols, values, n_samples, n_features)
    
    if tfidf.norm !== nothing
        X = normalize(X, tfidf.norm)
    end
    
    return X
end

function _transform(tfidf::TfidfVectorizer, X::AbstractMatrix)
    if tfidf.sublinear_tf
        X = X.nzval .= 1.0 .+ log.(X.nzval)
    end
    
    if tfidf.use_idf
        X = X * spdiagm(0 => tfidf.idf_)
    end
    
    if tfidf.norm !== nothing
        X = normalize(X, tfidf.norm)
    end
    
    return X
end

Base.getproperty(tfidf::TfidfVectorizer, sym::Symbol) = 
    sym === :vocabulary ? getfield(tfidf, :cv).vocabulary_ : getfield(tfidf, sym)

Base.setproperty!(tfidf::TfidfVectorizer, sym::Symbol, value) =
    sym === :vocabulary ? setfield!(getfield(tfidf, :cv), :vocabulary_, value) : setfield!(tfidf, sym, value)

function normalize(X::AbstractMatrix, norm::String)
    if norm == "l1"
        return X ./ sum(abs.(X), dims=2)
    elseif norm == "l2"
        return X ./ sqrt.(sum(X.^2, dims=2))
    else
        throw(ArgumentError("Invalid norm. Use 'l1' or 'l2'."))
    end
end
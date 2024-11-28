using SparseArrays

mutable struct CountVectorizer
    input::String
    encoding::String
    decode_error::String
    strip_accents::Union{Nothing,String,Function}
    lowercase::Bool
    preprocessor::Union{Nothing,Function}
    tokenizer::Union{Nothing,Function}
    stop_words::Union{Nothing,String,Vector{String},Set{String}}
    token_pattern::String
    ngram_range::Tuple{Int,Int}
    analyzer::Union{String,Function}
    max_df::Union{Float64,Int}
    min_df::Union{Float64,Int}
    max_features::Union{Nothing,Int}
    binary::Bool
    dtype::DataType

    # Fitted attributes
    vocabulary_::Dict{String,Int}
    stop_words_set_::Set{String}  # New field to store processed stop words
    fitted::Bool

    function CountVectorizer(;
        input::String="content",
        encoding::String="utf-8",
        decode_error::String="strict",
        strip_accents::Union{Nothing,String,Function}=nothing,
        lowercase::Bool=true,
        preprocessor::Union{Nothing,Function}=nothing,
        tokenizer::Union{Nothing,Function}=nothing,
        stop_words::Union{Nothing,String,Vector{String},Set{String}}=nothing,
        token_pattern::String="\\b\\w+\\b",
        ngram_range::Tuple{Int,Int}=(1, 1),
        analyzer::Union{String,Function}="word",
        max_df::Union{Float64,Int}=1.0,
        min_df::Union{Float64,Int}=1,
        max_features::Union{Nothing,Int}=nothing,
        binary::Bool=false,
        dtype::DataType=Int64
    )
        # Initialize stop words set
        stop_words_set = if stop_words === nothing
            Set{String}()
        elseif stop_words isa String && stop_words == "english"
            # common English stop words
            Set([
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                "to", "was", "were", "will", "with", "the", "this", "but", "they",
                "have", "had", "what", "when", "where", "who", "which", "why", "how"
            ])
        elseif stop_words isa Vector{String}
            Set(stop_words)
        elseif stop_words isa Set{String}
            stop_words
        else
            Set{String}()
        end

        new(input, encoding, decode_error, strip_accents, lowercase,
            preprocessor, tokenizer, stop_words, token_pattern,
            ngram_range, analyzer, max_df, min_df, max_features,
            binary, dtype, Dict{String,Int}(), stop_words_set, false)
    end
end

function (cv::CountVectorizer)(raw_documents::AbstractVector{T} where {T<:AbstractString})
    cv.vocabulary_ = _build_vocabulary(cv, raw_documents)
    cv.fitted = true
    result = _transform(cv, raw_documents)
    return result
end

function (cv::CountVectorizer)(X::AbstractMatrix; type::Symbol=:transform)
    if type == :inverse
        return _inverse_transform(cv, X)
    elseif type == :transform
        error("Use CountVectorizer(raw_documents) for transforming raw documents.")
    else
        throw(ArgumentError("Invalid type. Use :inverse for inverse transformation."))
    end
end

function _build_analyzer(cv::CountVectorizer)
    if isa(cv.analyzer, Function)
        return cv.analyzer
    elseif cv.analyzer == "word"
        return doc -> _word_ngrams(cv, _tokenize(cv, preprocess(cv, doc)))
    elseif cv.analyzer == "char"
        return doc -> _char_ngrams(cv, preprocess(cv, doc))
    elseif cv.analyzer == "char_wb"
        return doc -> _char_wb_ngrams(cv, preprocess(cv, doc))
    else
        throw(ArgumentError("Invalid analyzer"))
    end
end

function preprocess(cv::CountVectorizer, doc::AbstractString)
    if cv.preprocessor !== nothing
        doc = cv.preprocessor(doc)
    end
    if cv.lowercase
        doc = lowercase(doc)
    end
    if cv.strip_accents !== nothing
        if isa(cv.strip_accents, Function)
            doc = cv.strip_accents(doc)
        elseif cv.strip_accents == "ascii"
            doc = Unicode.normalize(doc, stripmark=true)
        elseif cv.strip_accents == "unicode"
            doc = Unicode.normalize(doc, :NFKD)
        end
    end
    return String(doc)  # Convert to String to ensure consistency
end

function _tokenize(cv::CountVectorizer, doc::AbstractString)
    if cv.tokenizer !== nothing
        tokens = cv.tokenizer(doc)
    else
        tokens = String[lowercase(String(m.match)) for m in eachmatch(r"\b\w+\b", doc)]
    end
    return tokens
end

function _word_ngrams(cv::CountVectorizer, tokens::Vector{T}) where {T<:AbstractString}
    # Filter stop words using the pre-computed set
    filtered_tokens = if !isempty(cv.stop_words_set_)
        filter(token -> !(token in cv.stop_words_set_), tokens)
    else
        tokens
    end

    min_n, max_n = cv.ngram_range
    ngrams = String[]

    if max_n == 1
        append!(ngrams, filtered_tokens)
    else
        for n in min_n:max_n
            for i in 1:(length(filtered_tokens)-n+1)
                push!(ngrams, join(filtered_tokens[i:(i+n-1)], " "))
            end
        end
    end
    return ngrams
end

function _build_vocabulary(cv::CountVectorizer, raw_documents::AbstractVector{T} where {T<:AbstractString})
    analyzer = _build_analyzer(cv)
    vocabulary = Dict{String,Int}()
    document_counts = Dict{String,Int}()

    all_terms = Set{String}()
    for (i, doc) in enumerate(raw_documents)
        doc_terms = analyzer(doc)

        if !(doc_terms isa AbstractVector)
            error("Analyzer returned $(typeof(doc_terms)), expected a vector of strings")
        end

        union!(all_terms, doc_terms)
        for term in doc_terms
            if !(term isa AbstractString)
                error("Expected string token, got $(typeof(term)): $term")
            end
            document_counts[term] = get(document_counts, term, 0) + 1
        end
    end

    # Create vocabulary with all unique terms
    for (index, term) in enumerate(sort(collect(all_terms)))
        vocabulary[term] = index
    end

    if isempty(vocabulary)
        error("Empty vocabulary. Check your analyzer and input documents.")
    end
    return vocabulary
end

function _char_ngrams(cv::CountVectorizer, text::AbstractString)
    min_n, max_n = cv.ngram_range
    ngrams = String[]
    for n in min_n:max_n
        for i in 1:(length(text)-n+1)
            push!(ngrams, String(text[i:(i+n-1)]))
        end
    end
    return ngrams
end

function _char_wb_ngrams(cv::CountVectorizer, text::AbstractString)
    min_n, max_n = cv.ngram_range
    ngrams = String[]
    for n in min_n:max_n
        for word in split(text)
            if length(word) > n
                for i in 1:(length(word)-n+1)
                    push!(ngrams, String(word[i:(i+n-1)]))
                end
            end
        end
    end
    return ngrams
end

function _transform(cv::CountVectorizer, raw_documents::AbstractVector{T} where {T<:AbstractString})
    analyzer = _build_analyzer(cv)
    n_samples = length(raw_documents)
    n_features = length(cv.vocabulary_)

    rows = Int[]
    cols = Int[]
    values = Int[]

    for (doc_idx, doc) in enumerate(raw_documents)
        feature_counter = Dict{Int,Int}()
        doc_terms = analyzer(doc)
        for term in doc_terms
            feature_idx = get(cv.vocabulary_, term, 0)
            if feature_idx != 0
                feature_counter[feature_idx] = get(feature_counter, feature_idx, 0) + 1
            else
                println("Warning: Term '$term' not found in vocabulary")
            end
        end

        for (feature_idx, count) in feature_counter
            push!(rows, doc_idx)
            push!(cols, feature_idx)
            push!(values, cv.binary ? 1 : count)
        end
    end

    if isempty(rows) || isempty(cols) || isempty(values)
        return spzeros(n_samples, n_features)
    end

    if any(col -> col < 1 || col > n_features, cols)
        invalid_cols = filter(col -> col < 1 || col > n_features, cols)
        error("Invalid column indices found: $invalid_cols. Number of features: $n_features")
    end

    X = sparse(rows, cols, values, n_samples, n_features)
    return X
end

function _inverse_transform(cv::CountVectorizer, X::AbstractMatrix)
    if !cv.fitted
        throw(ErrorException("CountVectorizer is not fitted"))
    end

    n_samples, n_features = size(X)
    if n_features != length(cv.vocabulary_)
        throw(DimensionMismatch("Shape of X does not match vocabulary size"))
    end

    inverse_vocabulary = Dict(idx => term for (term, idx) in cv.vocabulary_)

    return [
        join([inverse_vocabulary[j] for j in 1:n_features if X[i, j] != 0], " ")
        for i in 1:n_samples
    ]
end

Base.getproperty(cv::CountVectorizer, sym::Symbol) =
    sym === :vocabulary ? getfield(cv, :vocabulary_) : getfield(cv, sym)

Base.setproperty!(cv::CountVectorizer, sym::Symbol, value) =
    sym === :vocabulary ? setfield!(cv, :vocabulary_, value) : setfield!(cv, sym, value)
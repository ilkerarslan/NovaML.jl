using SparseArrays

mutable struct CountVectorizer
    input::String
    encoding::String
    decode_error::String
    strip_accents::Union{Nothing, String, Function}
    lowercase::Bool
    preprocessor::Union{Nothing, Function}
    tokenizer::Union{Nothing, Function}
    stop_words::Union{Nothing, String, Vector{String}}
    token_pattern::String
    ngram_range::Tuple{Int, Int}
    analyzer::Union{String, Function}
    max_df::Union{Float64, Int}
    min_df::Union{Float64, Int}
    max_features::Union{Nothing, Int}
    binary::Bool
    dtype::DataType
    
    # Fitted attributes
    vocabulary_::Dict{String, Int}
    fitted::Bool

    function CountVectorizer(;
        input::String="content",
        encoding::String="utf-8",
        decode_error::String="strict",
        strip_accents::Union{Nothing, String, Function}=nothing,
        lowercase::Bool=true,
        preprocessor::Union{Nothing, Function}=nothing,
        tokenizer::Union{Nothing, Function}=nothing,
        stop_words::Union{Nothing, String, Vector{String}}=nothing,
        token_pattern::String="\\b\\w+\\b",
        ngram_range::Tuple{Int, Int}=(1, 1),
        analyzer::Union{String, Function}="word",
        max_df::Union{Float64, Int}=1.0,
        min_df::Union{Float64, Int}=1,
        max_features::Union{Nothing, Int}=nothing,
        binary::Bool=false,
        dtype::DataType=Int64
    )
        new(input, encoding, decode_error, strip_accents, lowercase,
            preprocessor, tokenizer, stop_words, token_pattern,
            ngram_range, analyzer, max_df, min_df, max_features,
            binary, dtype, Dict{String, Int}(), false)
    end
end

function (cv::CountVectorizer)(raw_documents::Vector{String})
    println("CountVectorizer called with $(length(raw_documents)) documents")
    println("Initial state: fitted=$(cv.fitted), vocabulary_=$(cv.vocabulary_)")
    
    cv.vocabulary_ = _build_vocabulary(cv, raw_documents)
    cv.fitted = true
    
    println("After building vocabulary: vocabulary_=$(cv.vocabulary_)")
    
    result = _transform(cv, raw_documents)
    println("Transform result shape: $(size(result))")
    println("Transform result nonzeros: $(nnz(result))")
    println("Transform result full: $(Matrix(result))")
    
    return result
end

function (cv::CountVectorizer)(X::AbstractMatrix; type::Symbol=:transform)
    if type == :inverse_transform
        return _inverse_transform(cv, X)
    elseif type == :transform
        error("Use CountVectorizer(raw_documents) for transforming raw documents.")
    else
        throw(ArgumentError("Invalid type. Use :inverse_transform for inverse transformation."))
    end
end

function _build_analyzer(cv::CountVectorizer)
    println("Building analyzer")
    if isa(cv.analyzer, Function)
        println("Using custom analyzer function")
        return cv.analyzer
    elseif cv.analyzer == "word"
        println("Using word analyzer")
        return doc -> _word_ngrams(cv, _tokenize(cv, preprocess(cv, doc)))
    elseif cv.analyzer == "char"
        println("Using char analyzer")
        return doc -> _char_ngrams(cv, preprocess(cv, doc))
    elseif cv.analyzer == "char_wb"
        println("Using char_wb analyzer")
        return doc -> _char_wb_ngrams(cv, preprocess(cv, doc))
    else
        throw(ArgumentError("Invalid analyzer"))
    end
end

function preprocess(cv::CountVectorizer, doc::String)
    println("Preprocessing: $doc")
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
    println("Preprocessed: $doc")
    return doc
end

function _tokenize(cv::CountVectorizer, doc::String)
    println("Tokenizing: $doc")
    if cv.tokenizer !== nothing
        tokens = cv.tokenizer(doc)
    else
        # Use a regex that captures all words, including "is" and "the"
        tokens = String[lowercase(m.match) for m in eachmatch(r"\b\w+\b", doc)]
    end
    println("Tokens: $tokens")
    return tokens
end

function _word_ngrams(cv::CountVectorizer, tokens::Vector{T}) where T <: AbstractString
    println("Generating word n-grams from tokens: $tokens")
    if cv.stop_words !== nothing
        tokens = [token for token in tokens if token ∉ cv.stop_words]
    end
    
    min_n, max_n = cv.ngram_range
    ngrams = String[]
    
    if max_n == 1
        append!(ngrams, tokens)
    else
        for n in min_n:max_n
            for i in 1:(length(tokens) - n + 1)
                push!(ngrams, join(tokens[i:(i+n-1)], " "))
            end
        end
    end
    
    println("Generated n-grams: $ngrams")
    return ngrams
end

function _build_vocabulary(cv::CountVectorizer, raw_documents::Vector{String})
    println("Entering _build_vocabulary")
    
    analyzer = _build_analyzer(cv)
    vocabulary = Dict{String, Int}()
    document_counts = Dict{String, Int}()
    
    println("Analyzing documents")
    all_terms = Set{String}()
    for (i, doc) in enumerate(raw_documents)
        println("Processing document $i: $doc")
        doc_terms = analyzer(doc)
        
        if !(doc_terms isa AbstractVector)
            error("Analyzer returned $(typeof(doc_terms)), expected a vector of strings")
        end
        
        println("Terms in document $i: $doc_terms")
        union!(all_terms, doc_terms)
        for term in doc_terms
            if !(term isa AbstractString)
                error("Expected string token, got $(typeof(term)): $term")
            end
            document_counts[term] = get(document_counts, term, 0) + 1
        end
    end
    
    println("Document counts: $document_counts")
    println("All unique terms: $all_terms")
    
    # Create vocabulary with all unique terms
    for (index, term) in enumerate(sort(collect(all_terms)))
        vocabulary[term] = index
    end
    
    println("Final vocabulary: $vocabulary")
    if isempty(vocabulary)
        error("Empty vocabulary. Check your analyzer and input documents.")
    end
    return vocabulary
end

function _char_ngrams(cv::CountVectorizer, text::String)
    min_n, max_n = cv.ngram_range
    ngrams = String[]
    for n in min_n:max_n
        for i in 1:(length(text) - n + 1)
            push!(ngrams, text[i:(i+n-1)])
        end
    end
    return ngrams
end

function _char_wb_ngrams(cv::CountVectorizer, text::String)
    min_n, max_n = cv.ngram_range
    ngrams = String[]
    for n in min_n:max_n
        for word in split(text)
            if length(word) > n
                for i in 1:(length(word) - n + 1)
                    push!(ngrams, word[i:(i+n-1)])
                end
            end
        end
    end
    return ngrams
end

function _transform(cv::CountVectorizer, raw_documents::Vector{String})
    analyzer = _build_analyzer(cv)
    n_samples = length(raw_documents)
    n_features = length(cv.vocabulary_)
    
    rows = Int[]
    cols = Int[]
    values = Int[]
    
    println("Vocabulary: $(cv.vocabulary_)")
    println("Number of features: $n_features")

    for (doc_idx, doc) in enumerate(raw_documents)
        feature_counter = Dict{Int, Int}()
        doc_terms = analyzer(doc)
        println("Document $doc_idx terms: $doc_terms")
        for term in doc_terms
            feature_idx = get(cv.vocabulary_, term, 0)
            if feature_idx != 0
                feature_counter[feature_idx] = get(feature_counter, feature_idx, 0) + 1
            else
                println("Warning: Term '$term' not found in vocabulary")
            end
        end
        
        println("Document $doc_idx features: $feature_counter")
        
        for (feature_idx, count) in feature_counter
            push!(rows, doc_idx)
            push!(cols, feature_idx)
            push!(values, cv.binary ? 1 : count)
        end
    end
    
    println("Rows: $rows")
    println("Columns: $cols")
    println("Values: $values")

    if isempty(rows) || isempty(cols) || isempty(values)
        println("Warning: Empty matrix data. Returning zero matrix.")
        return spzeros(n_samples, n_features)
    end

    # Check if all column indices are within the correct range
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
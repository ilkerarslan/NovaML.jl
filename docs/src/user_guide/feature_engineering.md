# Feature Engineering

Feature engineering is the process of using domain knowledge to create new features or transform existing ones to improve machine learning model performance. NovaML provides several tools to help you effectively engineer and select features for your models.

## Polynomial Features

One common technique for capturing non-linear relationships in your data is to create polynomial and interaction features. NovaML's `PolynomialFeatures` transformer can automatically generate these higher-order features.

```julia
using NovaML.PreProcessing

# Create a transformer that will generate polynomial features up to degree 2
poly = PolynomialFeatures(degree=2)

# Example data with two features
X = [1 2; 3 4; 5 6]

# Generate polynomial features
X_poly = poly(X)

# The output will include: 
# - Original features (x₁, x₂)
# - Squared terms (x₁², x₂²)
# - Interaction terms (x₁x₂)
```

You can control the complexity of the generated features with various parameters:

```julia
# Generate only interaction terms, without higher-order terms
poly_interact = PolynomialFeatures(
    degree=2, 
    interaction_only=true
)

# Generate features up to degree 3, including bias term
poly_cubic = PolynomialFeatures(
    degree=3, 
    include_bias=true
)
```

## Text Feature Extraction

NovaML provides tools for converting text data into numerical features that can be used by machine learning algorithms.

### Count Vectorization

The `CountVectorizer` transforms text documents into a matrix of token counts:

```julia
using NovaML.FeatureExtraction

# Example text documents
docs = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The mat was red"
]

countvec = CountVectorizer();
bag = countvec(docs);
countvec.vocabulary
countvec(bag, type=:inverse)
```

You can customize the vectorization process with various parameters:

```julia
vectorizer = CountVectorizer(
    min_df=2,               # Ignore terms that appear in less than 2 documents
    max_df=0.95,           # Ignore terms that appear in more than 95% of documents
    stop_words="english",   # Remove common English stop words
    ngram_range=(1, 2)     # Include both unigrams and bigrams
)
```

### TF-IDF Vectorization

The `TfidfVectorizer` converts a collection of raw documents to a matrix of TF-IDF features:

```julia
# Initialize and fit TF-IDF vectorizer
tfidf = TfidfVectorizer()
X_tfidf = tfidf(documents)

# Transform new documents using the fitted vectorizer
new_docs = ["A cat and dog play", "The red mat"]
X_new = tfidf(new_docs)
```

## Feature Selection

NovaML helps you identify and select the most important features for your models. Feature selection can help improve model performance, reduce overfitting, and speed up training.

### Using Model-Based Feature Importance

Many models in NovaML provide feature importance scores that you can use for feature selection:

```julia
using NovaML.Tree

# Train a decision tree
dt = DecisionTreeClassifier(max_depth=5)
dt(X_train, y_train)

# Get feature importances
importances = dt.feature_importances_

# Print feature importances with their names
for (name, importance) in zip(feature_names, importances)
    println("$name: $importance")
end
```

## Combining Feature Engineering Steps

You can combine multiple feature engineering steps using NovaML's pipeline functionality:

```julia
using NovaML.Pipelines
using NovaML.PreProcessing

# Create a pipeline that:
# 1. Scales the numerical features
# 2. Generates polynomial features
# 3. Selects the most important features
# 4. Trains a model
pipe = pipe(
    StandardScaler(),
    PolynomialFeatures(degree=2),
    LogisticRegression()
)

# Fit the entire pipeline
pipe(X_train, y_train)

# Make predictions using the pipeline
y_pred = pipe(X_test)
```
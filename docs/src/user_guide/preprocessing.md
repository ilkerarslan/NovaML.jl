# Data Preprocessing

Data preprocessing is a crucial step in any machine learning pipeline. NovaML.jl provides a range of tools for cleaning, transforming, and preparing your data for model training. This page covers the main preprocessing techniques available in NovaML.

## Scaling

Scaling features is often necessary to ensure that all features contribute equally to the model training process. NovaML offers several scaling methods:

### StandardScaler

Standardizes features by removing the mean and scaling to unit variance.

```julia
using NovaML.Datasets
iris = load_iris()
X = iris["data"][:, 3:4]
y = iris["target"]
```
Split data to train and test sets.

```julia
using NovaML.ModelSelection
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, stratify=y)
```

Fit and transform `StandardScaler`.

```julia
using NovaML.PreProcessing

stdscaler = StandardScaler()

# fit and transform
Xtrnstd = stdscaler(Xtrn)

# transform
Xtststd = stdscaler(Xtst)

# inverse transform
Xtrn = stdscaler(Xtrnstd, type=:inverse)
```

### MinMaxScaler

Scales features to a number between [0, 1].

```julia
minmax = MinMaxScaler()

# fit & transform
Xtrn_mm = minmax(Xtrn)
# transform
Xtst_mm = minmax(Xtst)
# inverse transform
Xtrn = minmax(Xtrn_mm, type=:inverse)
```

## Encoding

Categorical variables often need to be encoded into numerical form for machine learning algorithms.

### LabelEncoder

Encodes target labels with value between 0 and n_classes-1.

```julia
using NovaML.PreProcessing

lblencode = LabelEncoder()

labels = ["M", "L", "XL", "M", "L", "M"]

# Label encode labels
labels = lblencode(labels)

# Get the labels back
lblencode(labels, :inverse)
```

### OneHotEncoder

Encodes categorical features as a one-hot numeric array.

```julia
using NovaML.PreProcessing

labels = ["M", "L", "XL", "M", "L", "M"]

ohe = OneHotEncoder()
onehot = ohe(labels)
ohe(onehot, :inverse)
```

### PolynomialFeatures

Generates polynomial and interaction features.

```julia
using NovaML.PreProcessing

X = rand(5, 2)
# 5×2 Matrix{Float64}:
#  0.85245   0.405935
#  0.139957  0.380467
#  0.730332  0.0418465
#  0.051091  0.570372
#  0.730245  0.128763

poly = PolynomialFeatures(
    degree=2,
    interaction_only=false,
    include_bias=true)

Xnew = poly(X)
# 5×6 Matrix{Float64}:
#  1.0  0.85245   0.405935   0.72667     0.346039   0.164783
#  1.0  0.139957  0.380467   0.0195881   0.0532491  0.144755
#  1.0  0.730332  0.0418465  0.533384    0.0305618  0.00175113
#  1.0  0.051091  0.570372   0.00261029  0.0291409  0.325324
#  1.0  0.730245  0.128763   0.533258    0.0940282  0.0165798
```

## Imputation
Missing data is a common issue in real-world datasets. NovaML provides tools for handling missing values. The `strategy` argument must be one of `:mean`, `:median`, `:most_frequent`, or `:constant`.

### SimpleImputer

Imputes missing values using a variety of strategies.

```julia
X = [1.0   2.0   3.0       4.0
     5.0   6.0    missing  8.0
    10.0  11.0  12.0        missing]

using NovaML.Impute
imputer = SimpleImputer(strategy=:mean)
Ximp = imputer(X)

# 3×4 Matrix{Union{Missing, Float64}}:
#   1.0   2.0   3.0  4.0
#   5.0   6.0   7.5  8.0
#  10.0  11.0  12.0  6.0
```

## Pipelines

You can combine multiple preprocessing steps into a single pipeline for easier management and application.

```julia
using NovaML.PreProcessing: StandardScaler
using NovaML.Decomposition: PCA
using NovaML.LinearModel: LogisticRegression

sc = StandardScaler()
pca = PCA(n_components=2)
lr = LogisticRegression()

# transform the data and fit the model 
Xtrn |> sc |> pca |> X -> lr(X, ytrn)

# make predictions
ŷtst = Xtst |> sc |> pca |> lr
```

It is also possible to create pipelines using NovaML's `Pipe` constructor:

```julia
using NovaML.Pipelines: pipe

# create a pipeline
pipe = pipe(
   StandardScaler(),
   PCA(n_components=2),
   LogisticRegression())

# fit the pipe
pipe(Xtrn, ytrn)
# make predictions
ŷ = pipe(Xtst) 
# make probability predictions
ŷprobs = pipe(Xtst, type=:probs)
```

## Text Preprocessing

For text data, NovaML offers vectorization techniques:

### CountVectorizer

Converts a collection of text documents to a matrix of token counts.

```julia
docs = [
    "Julia was designed for high performance",
    "Julia uses multiple dispatch as a paradigm",
    "Julia is dynamically typed, feels like a scripting language",
    "But can also optionally be separately compiled",
    "Julia is an open source project"];

using NovaML.FeatureExtraction

countvec = CountVectorizer();
bag = countvec(docs);
countvec.vocabulary

# Dict{String, Int64} with 30 entries:
#   "scripting"   => 25
#   "high"        => 14
#   "feels"       => 12
#   "is"          => 15
#   "separately"  => 26
#   "language"    => 17
#   "typed"       => 28
#   "but"         => 6
#   "a"           => 1
#   "for"         => 13
#   "optionally"  => 21
#   "paradigm"    => 22
#   "was"         => 30
#   "dynamically" => 11
#   "also"        => 2
#   "an"          => 3
#   "multiple"    => 19
#   "be"          => 5
#   "julia"       => 16
#   "project"     => 24
#   "uses"        => 29
#   "source"      => 27
#   "open"        => 20
#   "performance" => 23
#   "compiled"    => 8
#   "designed"    => 9
#   "as"          => 4
#   "can"         => 7
#   "like"        => 18
#   "dispatch"    => 10

countvec(bag, type=:inverse)
# 5-element Vector{String}:
#  "designed for high julia performance was"
#  "a as dispatch julia multiple paradigm uses"
#  "a dynamically feels is julia language like scripting typed"      
#  "also be but can compiled optionally separately"
#  "an is julia open project source"
```

### TfidfVectorizer

Converts a collection of raw documents to a matrix of TF-IDF features.

```julia
tfidf = TfidfVectorizer()

result = tfidf(docs)
tfidf.vocabulary
tfidf(result, type=:inverse)

new_docs = [" The talk on the Unreasonable Effectiveness of Multiple Dispatch explains why it works so well."]
Xnew = tfidf(new_docs)

```

Most preprocessing transforms in NovaML follow the functor pattern: transform(X) both fits the transformer to the data and applies the transformation. For separate fitting and transforming (e.g., when you want to apply the same transformation to test data), you can use the fitted transformer directly on new data.
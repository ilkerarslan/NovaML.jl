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

Scales features to a [0, 1].

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
encoder = LabelEncoder()
y_encoded = encoder(y)

# To invert the encoding:
y_original = encoder(y_encoded, type=:inverse_transform)
```

### OneHotEncoder

Encodes categorical features as a one-hot numeric array.

```julia
encoder = OneHotEncoder()
X_encoded = encoder(X)

# To invert the encoding:
X_original = encoder(X_encoded, type=:inverse_transform)
```

## Imputation
Missing data is a common issue in real-world datasets. NovaML provides tools for handling missing values.

### SimpleImputer

Imputes missing values using a variety of strategies.

```julia
using NovaML.Impute

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy=:mean)
X_imputed = imputer(X)

# Other strategies include :median, :most_frequent, and :constant
imputer_median = SimpleImputer(strategy=:median)
imputer_frequent = SimpleImputer(strategy=:most_frequent)
imputer_constant = SimpleImputer(strategy=:constant, fill_value=0)
```

## Feature Engineering

NovaML also provides tools for creating new features or transforming existing ones.

### PolynomialFeatures

Generates polynomial and interaction features.

```julia
poly = PolynomialFeatures(degree=2)
X_poly = poly(X)
```

## Pipelines

You can combine multiple preprocessing steps into a single pipeline for easier management and application. See the Pipelines section for more details.

```julia
using NovaML.Pipelines

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

## Text Preprocessing

For text data, NovaML offers vectorization techniques:

### CountVectorizer

Converts a collection of text documents to a matrix of token counts.

```julia
using NovaML.FeatureExtraction

vectorizer = CountVectorizer()
X_counts = vectorizer(documents)
```

### TfidfVectorizer

Converts a collection of raw documents to a matrix of TF-IDF features.

```julia
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer(documents)
```

Most preprocessing transforms in NovaML follow the functor pattern: transform(X) both fits the transformer to the data and applies the transformation. For separate fitting and transforming (e.g., when you want to apply the same transformation to test data), you can use the fitted transformer directly on new data.
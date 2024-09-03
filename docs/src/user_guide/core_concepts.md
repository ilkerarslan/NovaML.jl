# Core Concepts

NovaML.jl is designed with simplicity, flexibility, and performance in mind. Understanding the core concepts will help you make the most of the library.

## Functor-based API

One of the distinguishing features of NovaML is its use of functors (callable objects) for model training, prediction, and data transformation. This approach leverages Julia's multiple dispatch system to provide a clean and intuitive API.

### Models

For supervised learning models:

- `model(X, y)`: Trains the model on input data `X` and target values `y`.
- `model(X)`: Makes predictions on new data `X`.
- `model(X, type=:probs)`: Computes probability predictions (for classifiers).

For unsupervised learning models:

- `model(X)`: Fits the model to the data `X`.

### Transformers

For data preprocessing and feature engineering:

- `transformer(X)`: Fits the transformer to the data `X` and applies the transformation.
- `transformer(X, type=:inverse_transform)`: Applies the inverse transformation (if available).

## Abstract Types

NovaML uses a hierarchy of abstract types to organize its components:

- `AbstractModel`: Base type for all machine learning models.
- `AbstractMultiClass`: Subtype of `AbstractModel` for multi-class classifiers.
- `AbstractScaler`: Base type for scaling transformers.

These abstract types allow for easy extension and customization of the library.

## Unified API

NovaML strives to provide a consistent interface across different types of models and tasks. This unified API makes it easier to switch between different algorithms and encourages experimentation.

## Pipelines

NovaML supports the creation of machine learning pipelines, which allow you to chain multiple steps of data preprocessing and model training into a single object. Pipelines can be treated as models themselves, simplifying complex workflows.

## Hyperparameter Tuning

The library includes tools for automated hyperparameter tuning, such as grid search and random search. These can be easily integrated with cross-validation techniques to find optimal model configurations.

## Metrics and Evaluation

NovaML provides a range of metrics for evaluating model performance, as well as utilities for cross-validation and model selection.

## Data Handling

The library is designed to work seamlessly with Julia's native array types and supports both dense and sparse data structures.

## Modules and Methods

### Datasets

- `load_boston`: Loads the Boston Housing dataset, a classic regression problem. It contains information about housing in the Boston area, with 13 features and a target variable representing median home values.
- `load_iris`: Provides access to the famous Iris flower dataset, useful for classification tasks. It includes 150 samples with 4 features each, categorized into 3 different species of Iris.
- `load_breast_cancer`: Loads the Wisconsin Breast Cancer dataset, a binary classification problem. It contains features computed from digitized images of breast mass, with the goal of predicting whether a tumor is malignant or benign.
- `load_wine`: Offers the Wine recognition dataset, suitable for multi-class classification. It includes 13 features derived from chemical analysis of wines from three different cultivars in Italy.
- `make_blobs`: Generates isotropic Gaussian blobs for clustering or classification tasks. This function allows you to create synthetic datasets with a specified number of samples, features, and centers, useful for testing and benchmarking algorithms.
- `make_moons`: Generates a 2D binary classification dataset in the shape of two interleaving half moons. This synthetic dataset is ideal for visualizing and testing classification algorithms, especially those that can handle non-linear decision boundaries.

### PreProcessing

- `StandardScaler`: Standardize features by removing the mean and scaling to unit variance
- `MinMaxScaler`: Scale features to a given range
- `LabelEncoder`: Encode categorical features as integers
- `OneHotEncoder`: Encode categorical features as one-hot vectors
- `PolynomialFeatures`: Generate polynomial and interaction features up to a specified degree

### Impute

- `SimpleImputer`: A basic imputation transformer for filling in missing values in datasets using strategies such as mean, median, most frequent, or constant value.

### FeatureExtraction

- `CountVectorizer`: Convert a collection of text documents to a matrix of token counts, useful for text feature extraction
- `TfidfVectorizer`: Transform a collection of raw documents to a matrix of TF-IDF features, combining the functionality of `CountVectorizer` with TF-IDF weighting 

### Decomposition

- `LatentDirichletAllocation`: A generative statistical model that allows sets of observations to be explained by unobserved groups. It's commonly used for topic modeling in natural language processing.
- `PCA`: Principal Component Analysis, a dimensionality reduction technique that identifies the axes of maximum variance in high-dimensional data and projects it onto a lower-dimensional subspace.

### LinearModels

- `Adaline`: Adaptive Linear Neuron
- `ElasticNet`: Linear regression with combined L1 and L2 priors as regularizer, balancing between Lasso and Ridge models
- `Lasso`: Linear Model trained with L1 prior as regularizer, useful for producing sparse models
- `LinearRegression`: Linear regression algorithm
- `LogisticRegression`: Binary and multiclass logistic regression
- `Perceptron`: Simple perceptron algorithm
- `RANSACRegression`: Robust regression using Random Sample Consensus (RANSAC) algorithm. It's particularly effective for fitting models in the presence of significant outliers in the data.
- `Ridge`: Linear regression with L2 regularization, useful for dealing with multicollinearity in data

### MultiClass

- `MulticlassPerceptron`: An extension of the binary perceptron algorithm for multi-class classification problems. It learns a linear decision boundary for each class and updates weights based on misclassifications.
- `OneVsRestClassifier`: A strategy for multi-class classification that fits one binary classifier per class, treating the class as positive and all others as negative. It's versatile and can be used with any base binary classifier.

### Neighbors

- `KNeighborsClassifier`: K-nearest neighbors classifier

### SVM

- `SVC`: Support Vector Classifier. Binary classification which supports linear and RBF kernels. Doesn't support multiclass classification yet. 

### Tree

- `DecisionTreeClassifier`: Decision tree for classification
- `DecisionTreeRegressor`: Decision tree for regression

### Ensemble

- `AdaBoostClassifier`: An ensemble method that sequentially applies a base classifier to reweighted versions of the training data, giving more emphasis to incorrectly classified instances in subsequent iterations
- `BaggingClassifier`: A meta-estimator that fits base classifiers on random subsets of the original dataset and aggregates their predictions to form a final prediction.
- `GradientBoostingClassifier`: An ensemble method that builds an additive model in a forward stage-wise fashion, allowing for the optimization of arbitrary differentiable loss functions. It uses decision trees as base learners and combines them to create a strong predictive model.
- `RandomForestClassifier`: An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.
- `RandomForestRegressor`: An ensemble method that builds multiple decision trees for regression tasks and predicts by averaging their outputs. It combines bagging with random feature selection to create a robust, accurate model that often resists overfitting.
- `VotingClassifier`: A classifier that combines multiple machine learning classifiers and uses a majority vote or the average predicted probabilities to predict the class labels.

### Cluster

- `AgglomerativeClustering`: A hierarchical clustering algorithm that builds nested clusters by merging or splitting them successively. This bottom-up approach is versatile and can create clusters of various shapes.
- `DBSCAN`: Density-Based Spatial Clustering of Applications with Noise, a density-based clustering algorithm that groups together points that are closely packed together, marking points that lie alone in low-density regions as outliers.
- `KMeans`: A popular and simple clustering algorithm that partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean (cluster centroid). It's efficient for large datasets but assumes spherical clusters of similar size.

### Metrics

- `accuracy_score`: Calculates the accuracy classification score, i.e., the proportion of correct predictions.
- `auc`: Computes the Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve, evaluating the overall performance of a binary classifier.
- `confusion_matrix`: Computes a confusion matrix to evaluate the accuracy of a classification. It shows the counts of true positive, false positive, true negative, and false negative predictions.
- `mean_absolute_error`, `mae`: Computes the average absolute difference between estimated and true values. This metric is robust to outliers and provides a linear measure of error. `mae` is an alias for `mean_absolute_error`.
- `mean_squared_error`, `mse`: Computes the average squared difference between estimated and true values. `mse` is an alias for `mean_squared_error`.
- `r2_score`: Calculates the coefficient of determination (R²), measuring how well future samples are likely to be predicted by the model.
- `adj_r2_score`: Computes the adjusted R² score, which accounts for the number of predictors in the model, penalizing unnecessary complexity.
- `f1_score`: Computes the F1 score, which is the harmonic mean of precision and recall, providing a balance between the two.
- `matthews_corcoef`: Calculates the Matthews correlation coefficient (MCC), a measure of the quality of binary classifications, considering all four confusion matrix categories.
- `precision_score`: Computes the precision score, which is the ratio of true positive predictions to the total predicted positives.
- `recall_score`: Computes the recall score, which is the ratio of true positive predictions to the total actual positives.
- `roc_auc_score`: Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC), providing an aggregate measure of classifier performance.
- `roc_curve`: Produces the values (fpr, tpr) to plot the Receiver Operating Characteristic (ROC) curve, showing the trade-off between true positive rate and false positive rate at various threshold settings.
- `silhouette_samples`: Computes the silhouette coefficient for each sample in a dataset, measuring how similar an object is to its own cluster compared to other clusters. This metric is useful for evaluating the quality of clustering algorithms.

### ModelSelection

- `cross_val_score`: Apply cross validation score
- `GridSearchCV`: Perform exhaustive search over specified parameter values for an estimator.
- `learning_curve`: Generate learning curves to evaluate model performance as a function of the number of training samples, helping to diagnose bias and variance problems
- `RandomSearchCV`: Perform randomized search over specified parameter distributions for an estimator. RandomSearchCV is often more efficient than GridSearchCV for hyperparameter optimization, especially when the parameter space is large or when some parameters are more important than others.
- `StratifiedKFold`: Provides stratified k-fold cross-validator, ensuring that the proportion of samples for each class is roughly the same in each fold
- `train_test_split`: Split arrays or matrices into random train and test subsets
- `validation_curve`: Determine training and validation scores for varying parameter values, helping to assess how a model's performance changes with respect to a specific hyperparameter and aiding in hyperparameter tuning

### Pipelines

- `pipe`: NovaML supports piped data transformation and model training via `|>` operator or NovaML.Pipelines.pipe 
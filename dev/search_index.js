var documenterSearchIndex = {"docs":
[{"location":"user_guide/getting_started/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"user_guide/getting_started/#Installation","page":"Getting Started","title":"Installation","text":"","category":"section"},{"location":"user_guide/getting_started/","page":"Getting Started","title":"Getting Started","text":"You can install NovaML.jl using Julia's package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:","category":"page"},{"location":"user_guide/getting_started/","page":"Getting Started","title":"Getting Started","text":"pkg> add NovaML","category":"page"},{"location":"user_guide/getting_started/#Usage","page":"Getting Started","title":"Usage","text":"","category":"section"},{"location":"user_guide/getting_started/","page":"Getting Started","title":"Getting Started","text":"The most prominent feature of NovaML is using functors (callable objects) to keep parameters as well as training and prediction. Assume model represents a supervised algorithm. The struct model keeps learned parameters and hyperparameters. It also behave as a function. ","category":"page"},{"location":"user_guide/getting_started/","page":"Getting Started","title":"Getting Started","text":"model(X, y) trains the model. \nmodel(Xnew) calculates the predictions for Xnew. ","category":"page"},{"location":"user_guide/getting_started/","page":"Getting Started","title":"Getting Started","text":"Here's a quick example of how to use NovaML.jl for a binary classification task:","category":"page"},{"location":"user_guide/getting_started/","page":"Getting Started","title":"Getting Started","text":"# Import the Iris dataset from NovaML's Datasets module\nusing NovaML.Datasets\nX, y = load_iris(return_X_y=true)\n\n# Split the data into training and test sets\n# 80% for training, 20% for testing\nusing NovaML.ModelSelection\nXtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)\n\n# Import the StandardScaler for feature scaling\nusing NovaML.PreProcessing\nscaler = StandardScaler()\nscaler.fitted # false - the scaler is not yet fitted to any data\n\n# Fit the scaler to the training data and transform it\n# NovaML uses a functor approach, so calling scaler(Xtrn) both fits and transforms\nXtrnstd = scaler(Xtrn) \n\n# Transform the test data using the fitted scaler\nXtststd = scaler(Xtst)\n\n# Import LogisticRegression from LinearModel module\nusing NovaML.LinearModel\n\n# Create a LogisticRegression model with learning rate 0.1 and 100 iterations\nlr = LogisticRegression(η=0.1, num_iter=100)\n\n# Import OneVsRestClassifier for multi-class classification\nusing NovaML.MultiClass\n# Wrap the LogisticRegression model in a OneVsRestClassifier for multi-class support\novr = OneVsRestClassifier(lr)\n\n# Fit the OneVsRestClassifier model to the standardized training data\n# NovaML uses functors, so ovr(Xtrnstd, ytrn) fits the model\novr(Xtrnstd, ytrn)\n\n# Make predictions on training and test data\n# Calling ovr(X) makes predictions using the fitted model\nŷtrn = ovr(Xtrnstd)\nŷtst = ovr(Xtststd)\n\n# Import accuracy_score metric for model evaluation\nusing NovaML.Metrics\n\n# Calculate accuracy for training and test sets\nacc_trn = accuracy_score(ytrn, ŷtrn);\nacc_tst = accuracy_score(ytst, ŷtst);\n\n# Print the results\nprintln(\"Training accuracy: $acc_trn\")\nprintln(\"Test accuracy: $acc_tst\")\n# Output:\n# Training accuracy: 0.9833333333333333\n# Test accuracy: 0.9666666666666667","category":"page"},{"location":"user_guide/core_concepts/#Core-Concepts","page":"Core Concepts","title":"Core Concepts","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"NovaML.jl is designed with simplicity, flexibility, and performance in mind. Understanding the core concepts will help you make the most of the library.","category":"page"},{"location":"user_guide/core_concepts/#Functor-based-API","page":"Core Concepts","title":"Functor-based API","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"One of the distinguishing features of NovaML is its use of functors (callable objects) for model training, prediction, and data transformation. This approach leverages Julia's multiple dispatch system to provide a clean and intuitive API.","category":"page"},{"location":"user_guide/core_concepts/#Models","page":"Core Concepts","title":"Models","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"For supervised learning models:","category":"page"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"model(X, y): Trains the model on input data X and target values y.\nmodel(X): Makes predictions on new data X.\nmodel(X, type=:probs): Computes probability predictions (for classifiers).","category":"page"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"For unsupervised learning models:","category":"page"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"model(X): Fits the model to the data X.","category":"page"},{"location":"user_guide/core_concepts/#Transformers","page":"Core Concepts","title":"Transformers","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"For data preprocessing and feature engineering:","category":"page"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"transformer(X): Fits the transformer to the data X and applies the transformation.\ntransformer(X, type=:inverse_transform): Applies the inverse transformation (if available).","category":"page"},{"location":"user_guide/core_concepts/#Abstract-Types","page":"Core Concepts","title":"Abstract Types","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"NovaML uses a hierarchy of abstract types to organize its components:","category":"page"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"AbstractModel: Base type for all machine learning models.\nAbstractMultiClass: Subtype of AbstractModel for multi-class classifiers.\nAbstractScaler: Base type for scaling transformers.","category":"page"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"These abstract types allow for easy extension and customization of the library.","category":"page"},{"location":"user_guide/core_concepts/#Unified-API","page":"Core Concepts","title":"Unified API","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"NovaML strives to provide a consistent interface across different types of models and tasks. This unified API makes it easier to switch between different algorithms and encourages experimentation.","category":"page"},{"location":"user_guide/core_concepts/#Pipelines","page":"Core Concepts","title":"Pipelines","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"NovaML supports the creation of machine learning pipelines, which allow you to chain multiple steps of data preprocessing and model training into a single object. Pipelines can be treated as models themselves, simplifying complex workflows.","category":"page"},{"location":"user_guide/core_concepts/#Hyperparameter-Tuning","page":"Core Concepts","title":"Hyperparameter Tuning","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"The library includes tools for automated hyperparameter tuning, such as grid search and random search. These can be easily integrated with cross-validation techniques to find optimal model configurations.","category":"page"},{"location":"user_guide/core_concepts/#Metrics-and-Evaluation","page":"Core Concepts","title":"Metrics and Evaluation","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"NovaML provides a range of metrics for evaluating model performance, as well as utilities for cross-validation and model selection.","category":"page"},{"location":"user_guide/core_concepts/#Data-Handling","page":"Core Concepts","title":"Data Handling","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"The library is designed to work seamlessly with Julia's native array types and supports both dense and sparse data structures.","category":"page"},{"location":"user_guide/core_concepts/#Modules-and-Methods","page":"Core Concepts","title":"Modules and Methods","text":"","category":"section"},{"location":"user_guide/core_concepts/#Datasets","page":"Core Concepts","title":"Datasets","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"load_boston: Loads the Boston Housing dataset, a classic regression problem. It contains information about housing in the Boston area, with 13 features and a target variable representing median home values.\nload_iris: Provides access to the famous Iris flower dataset, useful for classification tasks. It includes 150 samples with 4 features each, categorized into 3 different species of Iris.\nload_breast_cancer: Loads the Wisconsin Breast Cancer dataset, a binary classification problem. It contains features computed from digitized images of breast mass, with the goal of predicting whether a tumor is malignant or benign.\nload_wine: Offers the Wine recognition dataset, suitable for multi-class classification. It includes 13 features derived from chemical analysis of wines from three different cultivars in Italy.\nmake_blobs: Generates isotropic Gaussian blobs for clustering or classification tasks. This function allows you to create synthetic datasets with a specified number of samples, features, and centers, useful for testing and benchmarking algorithms.\nmake_moons: Generates a 2D binary classification dataset in the shape of two interleaving half moons. This synthetic dataset is ideal for visualizing and testing classification algorithms, especially those that can handle non-linear decision boundaries.","category":"page"},{"location":"user_guide/core_concepts/#PreProcessing","page":"Core Concepts","title":"PreProcessing","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"StandardScaler: Standardize features by removing the mean and scaling to unit variance\nMinMaxScaler: Scale features to a given range\nLabelEncoder: Encode categorical features as integers\nOneHotEncoder: Encode categorical features as one-hot vectors\nPolynomialFeatures: Generate polynomial and interaction features up to a specified degree","category":"page"},{"location":"user_guide/core_concepts/#Impute","page":"Core Concepts","title":"Impute","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"SimpleImputer: A basic imputation transformer for filling in missing values in datasets using strategies such as mean, median, most frequent, or constant value.","category":"page"},{"location":"user_guide/core_concepts/#FeatureExtraction","page":"Core Concepts","title":"FeatureExtraction","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"CountVectorizer: Convert a collection of text documents to a matrix of token counts, useful for text feature extraction\nTfidfVectorizer: Transform a collection of raw documents to a matrix of TF-IDF features, combining the functionality of CountVectorizer with TF-IDF weighting ","category":"page"},{"location":"user_guide/core_concepts/#Decomposition","page":"Core Concepts","title":"Decomposition","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"LatentDirichletAllocation: A generative statistical model that allows sets of observations to be explained by unobserved groups. It's commonly used for topic modeling in natural language processing.\nPCA: Principal Component Analysis, a dimensionality reduction technique that identifies the axes of maximum variance in high-dimensional data and projects it onto a lower-dimensional subspace.","category":"page"},{"location":"user_guide/core_concepts/#LinearModels","page":"Core Concepts","title":"LinearModels","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"Adaline: Adaptive Linear Neuron\nElasticNet: Linear regression with combined L1 and L2 priors as regularizer, balancing between Lasso and Ridge models\nLasso: Linear Model trained with L1 prior as regularizer, useful for producing sparse models\nLinearRegression: Linear regression algorithm\nLogisticRegression: Binary and multiclass logistic regression\nPerceptron: Simple perceptron algorithm\nRANSACRegression: Robust regression using Random Sample Consensus (RANSAC) algorithm. It's particularly effective for fitting models in the presence of significant outliers in the data.\nRidge: Linear regression with L2 regularization, useful for dealing with multicollinearity in data","category":"page"},{"location":"user_guide/core_concepts/#MultiClass","page":"Core Concepts","title":"MultiClass","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"MulticlassPerceptron: An extension of the binary perceptron algorithm for multi-class classification problems. It learns a linear decision boundary for each class and updates weights based on misclassifications.\nOneVsRestClassifier: A strategy for multi-class classification that fits one binary classifier per class, treating the class as positive and all others as negative. It's versatile and can be used with any base binary classifier.","category":"page"},{"location":"user_guide/core_concepts/#Neighbors","page":"Core Concepts","title":"Neighbors","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"KNeighborsClassifier: K-nearest neighbors classifier","category":"page"},{"location":"user_guide/core_concepts/#SVM","page":"Core Concepts","title":"SVM","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"SVC: Support Vector Classifier. Binary classification which supports linear and RBF kernels. Doesn't support multiclass classification yet. ","category":"page"},{"location":"user_guide/core_concepts/#Tree","page":"Core Concepts","title":"Tree","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"DecisionTreeClassifier: Decision tree for classification\nDecisionTreeRegressor: Decision tree for regression","category":"page"},{"location":"user_guide/core_concepts/#Ensemble","page":"Core Concepts","title":"Ensemble","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"AdaBoostClassifier: An ensemble method that sequentially applies a base classifier to reweighted versions of the training data, giving more emphasis to incorrectly classified instances in subsequent iterations\nBaggingClassifier: A meta-estimator that fits base classifiers on random subsets of the original dataset and aggregates their predictions to form a final prediction.\nGradientBoostingClassifier: An ensemble method that builds an additive model in a forward stage-wise fashion, allowing for the optimization of arbitrary differentiable loss functions. It uses decision trees as base learners and combines them to create a strong predictive model.\nRandomForestClassifier: An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.\nRandomForestRegressor: An ensemble method that builds multiple decision trees for regression tasks and predicts by averaging their outputs. It combines bagging with random feature selection to create a robust, accurate model that often resists overfitting.\nVotingClassifier: A classifier that combines multiple machine learning classifiers and uses a majority vote or the average predicted probabilities to predict the class labels.","category":"page"},{"location":"user_guide/core_concepts/#Cluster","page":"Core Concepts","title":"Cluster","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"AgglomerativeClustering: A hierarchical clustering algorithm that builds nested clusters by merging or splitting them successively. This bottom-up approach is versatile and can create clusters of various shapes.\nDBSCAN: Density-Based Spatial Clustering of Applications with Noise, a density-based clustering algorithm that groups together points that are closely packed together, marking points that lie alone in low-density regions as outliers.\nKMeans: A popular and simple clustering algorithm that partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean (cluster centroid). It's efficient for large datasets but assumes spherical clusters of similar size.","category":"page"},{"location":"user_guide/core_concepts/#Metrics","page":"Core Concepts","title":"Metrics","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"accuracy_score: Calculates the accuracy classification score, i.e., the proportion of correct predictions.\nauc: Computes the Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve, evaluating the overall performance of a binary classifier.\nconfusion_matrix: Computes a confusion matrix to evaluate the accuracy of a classification. It shows the counts of true positive, false positive, true negative, and false negative predictions.\nmean_absolute_error, mae: Computes the average absolute difference between estimated and true values. This metric is robust to outliers and provides a linear measure of error. mae is an alias for mean_absolute_error.\nmean_squared_error, mse: Computes the average squared difference between estimated and true values. mse is an alias for mean_squared_error.\nr2_score: Calculates the coefficient of determination (R²), measuring how well future samples are likely to be predicted by the model.\nadj_r2_score: Computes the adjusted R² score, which accounts for the number of predictors in the model, penalizing unnecessary complexity.\nf1_score: Computes the F1 score, which is the harmonic mean of precision and recall, providing a balance between the two.\nmatthews_corcoef: Calculates the Matthews correlation coefficient (MCC), a measure of the quality of binary classifications, considering all four confusion matrix categories.\nprecision_score: Computes the precision score, which is the ratio of true positive predictions to the total predicted positives.\nrecall_score: Computes the recall score, which is the ratio of true positive predictions to the total actual positives.\nroc_auc_score: Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC), providing an aggregate measure of classifier performance.\nroc_curve: Produces the values (fpr, tpr) to plot the Receiver Operating Characteristic (ROC) curve, showing the trade-off between true positive rate and false positive rate at various threshold settings.\nsilhouette_samples: Computes the silhouette coefficient for each sample in a dataset, measuring how similar an object is to its own cluster compared to other clusters. This metric is useful for evaluating the quality of clustering algorithms.","category":"page"},{"location":"user_guide/core_concepts/#ModelSelection","page":"Core Concepts","title":"ModelSelection","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"cross_val_score: Apply cross validation score\nGridSearchCV: Perform exhaustive search over specified parameter values for an estimator.\nlearning_curve: Generate learning curves to evaluate model performance as a function of the number of training samples, helping to diagnose bias and variance problems\nRandomSearchCV: Perform randomized search over specified parameter distributions for an estimator. RandomSearchCV is often more efficient than GridSearchCV for hyperparameter optimization, especially when the parameter space is large or when some parameters are more important than others.\nStratifiedKFold: Provides stratified k-fold cross-validator, ensuring that the proportion of samples for each class is roughly the same in each fold\ntrain_test_split: Split arrays or matrices into random train and test subsets\nvalidation_curve: Determine training and validation scores for varying parameter values, helping to assess how a model's performance changes with respect to a specific hyperparameter and aiding in hyperparameter tuning","category":"page"},{"location":"user_guide/core_concepts/#Pipelines-2","page":"Core Concepts","title":"Pipelines","text":"","category":"section"},{"location":"user_guide/core_concepts/","page":"Core Concepts","title":"Core Concepts","text":"pipe: NovaML supports piped data transformation and model training via |> operator or NovaML.Pipelines.pipe ","category":"page"},{"location":"user_guide/preprocessing/#Data-Preprocessing","page":"Data Preprocessing","title":"Data Preprocessing","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Data preprocessing is a crucial step in any machine learning pipeline. NovaML.jl provides a range of tools for cleaning, transforming, and preparing your data for model training. This page covers the main preprocessing techniques available in NovaML.","category":"page"},{"location":"user_guide/preprocessing/#Scaling","page":"Data Preprocessing","title":"Scaling","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Scaling features is often necessary to ensure that all features contribute equally to the model training process. NovaML offers several scaling methods:","category":"page"},{"location":"user_guide/preprocessing/#StandardScaler","page":"Data Preprocessing","title":"StandardScaler","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Standardizes features by removing the mean and scaling to unit variance.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"using NovaML.Datasets\niris = load_iris()\nX = iris[\"data\"][:, 3:4]\ny = iris[\"target\"]","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Split data to train and test sets.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"using NovaML.ModelSelection\nXtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, stratify=y)","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Fit and transform StandardScaler.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"using NovaML.PreProcessing\n\nstdscaler = StandardScaler()\n\n# fit and transform\nXtrnstd = stdscaler(Xtrn)\n\n# transform\nXtststd = stdscaler(Xtst)\n\n# inverse transform\nXtrn = stdscaler(Xtrnstd, type=:inverse)","category":"page"},{"location":"user_guide/preprocessing/#MinMaxScaler","page":"Data Preprocessing","title":"MinMaxScaler","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Scales features to a number between [0, 1].","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"minmax = MinMaxScaler()\n\n# fit & transform\nXtrn_mm = minmax(Xtrn)\n# transform\nXtst_mm = minmax(Xtst)\n# inverse transform\nXtrn = minmax(Xtrn_mm, type=:inverse)","category":"page"},{"location":"user_guide/preprocessing/#Encoding","page":"Data Preprocessing","title":"Encoding","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Categorical variables often need to be encoded into numerical form for machine learning algorithms.","category":"page"},{"location":"user_guide/preprocessing/#LabelEncoder","page":"Data Preprocessing","title":"LabelEncoder","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Encodes target labels with value between 0 and n_classes-1.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"using NovaML.PreProcessing\n\nlblencode = LabelEncoder()\n\nlabels = [\"M\", \"L\", \"XL\", \"M\", \"L\", \"M\"]\n\n# Label encode labels\nlabels = lblencode(labels)\n\n# Get the labels back\nlblencode(labels, :inverse)","category":"page"},{"location":"user_guide/preprocessing/#OneHotEncoder","page":"Data Preprocessing","title":"OneHotEncoder","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Encodes categorical features as a one-hot numeric array.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"using NovaML.PreProcessing\n\nlabels = [\"M\", \"L\", \"XL\", \"M\", \"L\", \"M\"]\n\nohe = OneHotEncoder()\nonehot = ohe(labels)\nohe(onehot, :inverse)","category":"page"},{"location":"user_guide/preprocessing/#PolynomialFeatures","page":"Data Preprocessing","title":"PolynomialFeatures","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Generates polynomial and interaction features.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"poly = PolynomialFeatures(degree=2)\nX_poly = poly(X)","category":"page"},{"location":"user_guide/preprocessing/#Imputation","page":"Data Preprocessing","title":"Imputation","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Missing data is a common issue in real-world datasets. NovaML provides tools for handling missing values. The strategy argument must be one of :mean, :median, :most_frequent, or :constant.","category":"page"},{"location":"user_guide/preprocessing/#SimpleImputer","page":"Data Preprocessing","title":"SimpleImputer","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Imputes missing values using a variety of strategies.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"using NovaML.Impute\n\n# Impute missing values with the mean of the column\nimputer = SimpleImputer(strategy=:mean)\nX_imputed = imputer(X)","category":"page"},{"location":"user_guide/preprocessing/#Pipelines","page":"Data Preprocessing","title":"Pipelines","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"You can combine multiple preprocessing steps into a single pipeline for easier management and application. See the Pipelines section for more details.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"using NovaML.Pipelines\n\npipe = pipe(\n    StandardScaler(),\n    PolynomialFeatures(degree=2),\n    LogisticRegression()\n)\n\n# Fit the entire pipeline\npipe(X_train, y_train)\n\n# Make predictions using the pipeline\ny_pred = pipe(X_test)","category":"page"},{"location":"user_guide/preprocessing/#Text-Preprocessing","page":"Data Preprocessing","title":"Text Preprocessing","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"For text data, NovaML offers vectorization techniques:","category":"page"},{"location":"user_guide/preprocessing/#CountVectorizer","page":"Data Preprocessing","title":"CountVectorizer","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Converts a collection of text documents to a matrix of token counts.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"using NovaML.FeatureExtraction\n\nvectorizer = CountVectorizer()\nX_counts = vectorizer(documents)","category":"page"},{"location":"user_guide/preprocessing/#TfidfVectorizer","page":"Data Preprocessing","title":"TfidfVectorizer","text":"","category":"section"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Converts a collection of raw documents to a matrix of TF-IDF features.","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"vectorizer = TfidfVectorizer()\nX_tfidf = vectorizer(documents)","category":"page"},{"location":"user_guide/preprocessing/","page":"Data Preprocessing","title":"Data Preprocessing","text":"Most preprocessing transforms in NovaML follow the functor pattern: transform(X) both fits the transformer to the data and applies the transformation. For separate fitting and transforming (e.g., when you want to apply the same transformation to test data), you can use the fitted transformer directly on new data.","category":"page"},{"location":"#NovaML.jl","page":"Home","title":"NovaML.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"⚠️ IMPORTANT NOTE: NovaML.jl is currently in alpha stage. It is under active development and may contain bugs or incomplete features. Users should exercise caution and avoid using NovaML.jl in production environments at this time. We appreciate your interest and welcome feedback and contributions to help improve the package.","category":"page"},{"location":"","page":"Home","title":"Home","text":"NovaML.jl aims to provide a comprehensive and user-friendly machine learning framework written in Julia. Its objective is providing a unified API for various machine learning tasks, including supervised learning, unsupervised learning, and preprocessing, feature engineering etc.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Main objective of NovaML.jl is to increase the usage of Julia in daily data science and machine learning activities among students and practitioners.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Currently, the module and function naming in NovaML is similar to that of Scikit Learn to provide a familiarity to data science and machine learning practitioners. However, NovaML is not a wrapper of ScikitLearn.","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Unified API using Julia's multiple dispatch and functor-style callable objects\nAlgorithms for classification, regression, and clustering\nPreprocessing tools for data scaling, encoding, and imputation\nModel selection and evaluation utilities\nEnsemble methods","category":"page"},{"location":"models/clustering/#Clustering-Algorithms","page":"Clustering","title":"Clustering Algorithms","text":"","category":"section"},{"location":"models/clustering/#Agglomerative-Clustering","page":"Clustering","title":"Agglomerative Clustering","text":"","category":"section"},{"location":"models/clustering/#DBSCAN","page":"Clustering","title":"DBSCAN","text":"","category":"section"},{"location":"models/clustering/#KMeans","page":"Clustering","title":"KMeans","text":"","category":"section"}]
}

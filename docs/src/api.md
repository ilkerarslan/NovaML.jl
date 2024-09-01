# API Reference

## Datasets

```@docs
NovaML.Datasets.load_boston(; return_X_y=false)
NovaML.Datasets.load_iris(; return_X_y=false)
NovaML.Datasets.make_blobs(; kwargs...)
NovaML.Datasets.make_moons(; kwargs...)
NovaML.Datasets.load_wine(; return_X_y=false)
NovaML.Datasets.load_breast_cancer(; return_X_y=false)

```

## Clustering

### AgglomerativeClustering

```@docs
NovaML.Cluster.AgglomerativeClustering
NovaML.Cluster.AgglomerativeClustering(X::AbstractMatrix; y=nothing)
NovaML.Cluster.AgglomerativeClustering(X::AbstractMatrix, type::Symbol)
NovaML.Cluster.compute_distances
```

### DBSCAN

```@docs
NovaML.Cluster.DBSCAN
NovaML.Cluster.DBSCAN(X::AbstractMatrix, y=nothing; sample_weight=nothing)
NovaML.Cluster.get_params(dbscan::DBSCAN)
NovaML.Cluster.set_params!(dbscan::DBSCAN; kwargs...)
Base.show(io::IO, dbscan::DBSCAN)
```

### KMeans

```@docs
NovaML.Cluster.KMeans
NovaML.Cluster.KMeans(X::AbstractVecOrMat{Float64}, y=nothing; sample_weight=nothing)
NovaML.Cluster.initialize_centroids
NovaML.Cluster.kmeans_plus_plus
NovaML.Cluster.assign_labels
NovaML.Cluster.update_centroids
NovaML.Cluster.compute_inertia
NovaML.Cluster.get_params(kmeans::KMeans; deep=true)
NovaML.Cluster.set_params!(kmeans::KMeans; params...)
NovaML.Cluster.fit_predict
NovaML.Cluster.fit_transform
NovaML.Cluster.transform
NovaML.Cluster.score
Base.show(io::IO, kmeans::KMeans)
```

## Decomposition

### LatentDirichletAllocation

```@docs
NovaML.Decomposition.LatentDirichletAllocation
NovaML.Decomposition.LatentDirichletAllocation(X::AbstractMatrix{T}; type=nothing) where T <: Real
NovaML.Decomposition._fit_transform
NovaML.Decomposition._fit_batch
NovaML.Decomposition._fit_online
NovaML.Decomposition._e_step
NovaML.Decomposition._m_step
NovaML.Decomposition._transform
NovaML.Decomposition._perplexity
Base.show(io::IO, lda::LatentDirichletAllocation)
```

### PCA (Principla Component Analysis)
```@docs
NovaML.Decomposition.PCA
NovaML.Decomposition.PCA(X::AbstractMatrix{T}) where T <: Real
NovaML.Decomposition.PCA(X::AbstractMatrix{T}, mode::Symbol) where T <: Real
Base.show(io::IO, pca::PCA)
```

## Ensemble Methods

NovaML.Ensemble provides several ensemble learning methods for classification and regression tasks.

### AdaBoostClassifier

```@docs
NovaML.Ensemble.AdaBoostClassifier
NovaML.Ensemble.AdaBoostClassifier(X::AbstractMatrix, y::AbstractVector)
NovaML.Ensemble.AdaBoostClassifier(X::AbstractMatrix; type=nothing)
NovaML.Ensemble._compute_feature_importances(model::AdaBoostClassifier)
Base.show(io::IO, model::AdaBoostClassifier)
NovaML.Ensemble.get_params(model::AdaBoostClassifier; deep=true)
NovaML.Ensemble.set_params!(model::AdaBoostClassifier; kwargs...)
NovaML.Ensemble.decision_function(model::AdaBoostClassifier, X::AbstractMatrix)
NovaML.Ensemble.staged_predict(model::AdaBoostClassifier, X::AbstractMatrix)
NovaML.Ensemble.staged_predict_proba(model::AdaBoostClassifier, X::AbstractMatrix)
```

### BaggingClassifier

```@docs
NovaML.Ensemble.BaggingClassifier
NovaML.Ensemble.BaggingClassifier(X::AbstractMatrix, y::AbstractVector)
NovaML.Ensemble.BaggingClassifier(X::AbstractMatrix; type=nothing)
NovaML.Ensemble._compute_oob_score(bc::BaggingClassifier, X::AbstractMatrix, y::AbstractVector)
NovaML.Ensemble._generate_indices(bc::BaggingClassifier, n_samples::Int)
Base.show(io::IO, bc::BaggingClassifier)
```

### GradientBoostingClassifier

```@docs
NovaML.Ensemble.GradientBoostingClassifier
NovaML.Ensemble.GradientBoostingClassifier(X::AbstractMatrix, y::AbstractVector)
NovaML.Ensemble.GradientBoostingClassifier(X::AbstractMatrix; type=nothing)
NovaML.Ensemble.compute_negative_gradient
NovaML.Ensemble.compute_loss
NovaML.Ensemble.compute_feature_importances(gbm::GradientBoostingClassifier)
NovaML.Ensemble.InitialEstimator
NovaML.Ensemble.InitialEstimator(X::AbstractMatrix)
NovaML.Ensemble.fit_initial_estimator(y::AbstractVector)
NovaML.Ensemble.ZeroEstimator
NovaML.Ensemble.ZeroEstimator(X::AbstractMatrix)
Base.show(io::IO, gbm::GradientBoostingClassifier)
```

### RandomForestClassifier

```@docs
NovaML.Ensemble.RandomForestClassifier
NovaML.Ensemble.RandomForestClassifier(X::AbstractMatrix, y::AbstractVector)
NovaML.Ensemble.RandomForestClassifier(X::AbstractMatrix)
NovaML.Ensemble.get_max_features(forest::RandomForestClassifier, n_features::Int)
NovaML.Ensemble.bootstrap_sample(forest::RandomForestClassifier, X::AbstractMatrix, y::AbstractVector)
NovaML.Ensemble.calculate_tree_feature_importance(tree::DecisionTreeClassifier, feature_indices::Vector{Int}, n_features::Int)
Base.show(io::IO, forest::RandomForestClassifier)
```

### RandomForestRegressor

```@docs
NovaML.Ensemble.RandomForestRegressor
NovaML.Ensemble.RandomForestRegressor(X::AbstractMatrix, y::AbstractVector)
NovaML.Ensemble.RandomForestRegressor(X::AbstractMatrix)
NovaML.Ensemble.get_max_features(forest::RandomForestRegressor, n_features::Int)
NovaML.Ensemble.compute_oob_score(forest::RandomForestRegressor, X::AbstractMatrix, y::AbstractVector)
Base.show(io::IO, forest::RandomForestRegressor)
```

### VotingClassifier

```@docs
NovaML.Ensemble.VotingClassifier
NovaML.Ensemble.VotingClassifier(X::AbstractMatrix, y::AbstractVector)
NovaML.Ensemble.VotingClassifier(X::AbstractMatrix; type=nothing)
NovaML.Ensemble.transform(vc::VotingClassifier, X::AbstractMatrix)
Base.show(io::IO, vc::VotingClassifier)
```







# API Reference

## Clustering

NovaML.Cluster provides several clustering algorithms for unsupervised learning tasks.

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
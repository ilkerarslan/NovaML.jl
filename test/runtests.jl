using NovaML
using Test

@testset "NovaML.jl" begin

    @testset "KNeighborsRegressor" begin
        using NovaML.Neighbors: KNeighborsRegressor

        @testset "Constructor defaults" begin
            reg = KNeighborsRegressor()
            @test reg.n_neighbors == 5
            @test reg.weights == :uniform
            @test reg.algorithm == :auto
            @test reg.leaf_size == 30
            @test reg.p == 2
            @test reg.fitted == false
            @test reg.X === nothing
            @test reg.y === nothing
        end

        @testset "Constructor custom params" begin
            reg = KNeighborsRegressor(n_neighbors=3, weights=:distance, p=1, metric="manhattan")
            @test reg.n_neighbors == 3
            @test reg.weights == :distance
        end

        @testset "Constructor validation" begin
            @test_throws AssertionError KNeighborsRegressor(n_neighbors=0)
            @test_throws AssertionError KNeighborsRegressor(n_neighbors=-1)
            @test_throws AssertionError KNeighborsRegressor(weights=:invalid)
            @test_throws AssertionError KNeighborsRegressor(leaf_size=0)
            @test_throws AssertionError KNeighborsRegressor(p=0)
        end

        @testset "Fit" begin
            reg = KNeighborsRegressor(n_neighbors=2)
            X_train = [1.0 0.0; 2.0 0.0; 3.0 0.0; 4.0 0.0]
            y_train = [1.0, 2.0, 3.0, 4.0]
            reg(X_train, y_train)

            @test reg.fitted == true
            @test reg.n_features_in_ == 2
            @test reg.n_samples_fit_ == 4
            @test reg.X == X_train
            @test reg.y == y_train
        end

        @testset "Predict not fitted" begin
            reg = KNeighborsRegressor()
            @test_throws ErrorException reg(Float64[1.0 0.0])
        end

        @testset "Predict uniform weights" begin
            reg = KNeighborsRegressor(n_neighbors=2)
            X_train = [1.0 0.0; 2.0 0.0; 3.0 0.0; 4.0 0.0]
            y_train = [10.0, 20.0, 30.0, 40.0]
            reg(X_train, y_train)

            # Query point [1.5, 0.0] — two nearest are [1,0] (y=10) and [2,0] (y=20)
            preds = reg(Float64[1.5 0.0])
            @test length(preds) == 1
            @test preds[1] ≈ 15.0  # mean(10, 20)

            # Query point [3.5, 0.0] — two nearest are [3,0] (y=30) and [4,0] (y=40)
            preds = reg(Float64[3.5 0.0])
            @test preds[1] ≈ 35.0  # mean(30, 40)
        end

        @testset "Predict distance weights" begin
            reg = KNeighborsRegressor(n_neighbors=2, weights=:distance)
            X_train = [0.0 0.0; 10.0 0.0]
            y_train = [0.0, 10.0]
            reg(X_train, y_train)

            # Query point [1.0, 0.0]: dist to [0,0]=1, dist to [10,0]=9
            # w1 = 1/1 = 1, w2 = 1/9 ≈ 0.111
            # pred = (1*0 + 0.111*10) / (1 + 0.111) = 1.111/1.111 ≈ 1.0
            # (Using exact: w1=1/(1+eps), w2=1/(9+eps), result ≈ 1.0)
            preds = reg(Float64[1.0 0.0])
            @test preds[1] ≈ 1.0 atol=0.1

            # Query point [9.0, 0.0]: dist to [0,0]=9, dist to [10,0]=1
            # Closer to 10.0 → prediction should be near 9.0
            preds = reg(Float64[9.0 0.0])
            @test preds[1] ≈ 9.0 atol=0.1
        end

        @testset "Predict distance weights exact match" begin
            reg = KNeighborsRegressor(n_neighbors=2, weights=:distance)
            X_train = [0.0 0.0; 10.0 0.0]
            y_train = [5.0, 50.0]
            reg(X_train, y_train)

            # Query exactly on a training point: distance=0, w=1/eps() dominates
            preds = reg(Float64[0.0 0.0])
            @test preds[1] ≈ 5.0 atol=1e-10
        end

        @testset "Predict multiple samples" begin
            reg = KNeighborsRegressor(n_neighbors=1)
            X_train = [0.0 0.0; 1.0 0.0; 2.0 0.0]
            y_train = [100.0, 200.0, 300.0]
            reg(X_train, y_train)

            X_test = [0.1 0.0; 0.9 0.0; 2.1 0.0]
            preds = reg(X_test)
            @test length(preds) == 3
            @test preds[1] ≈ 100.0  # nearest to [0,0]
            @test preds[2] ≈ 200.0  # nearest to [1,0]
            @test preds[3] ≈ 300.0  # nearest to [2,0]
        end

        @testset "Fit returns self" begin
            reg = KNeighborsRegressor()
            X_train = [1.0 0.0; 2.0 0.0]
            y_train = [1.0, 2.0]
            result = reg(X_train, y_train)
            @test result === reg
        end

        @testset "Integer y converted to Float64" begin
            reg = KNeighborsRegressor(n_neighbors=1)
            X_train = [1.0 0.0; 2.0 0.0]
            y_train = [1, 2]  # integers
            reg(X_train, y_train)
            @test eltype(reg.y) == Float64
        end

        @testset "show method" begin
            reg = KNeighborsRegressor(n_neighbors=3)
            buf = IOBuffer()
            show(buf, reg)
            s = String(take!(buf))
            @test occursin("KNeighborsRegressor", s)
            @test occursin("n_neighbors=3", s)
            @test occursin("fitted=false", s)
        end
    end

    @testset "GradientBoostingRegressor" begin
        using NovaML.Ensemble: GradientBoostingRegressor
        using Random

        @testset "Constructor defaults" begin
            gbr = GradientBoostingRegressor()
            @test gbr.loss == "squared_error"
            @test gbr.learning_rate == 0.1
            @test gbr.n_estimators == 100
            @test gbr.subsample == 1.0
            @test gbr.max_depth == 3
            @test gbr.alpha == 0.9
            @test gbr.fitted == false
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError GradientBoostingRegressor(loss="invalid")
            @test_throws ArgumentError GradientBoostingRegressor(init="bad")
            @test_throws ArgumentError GradientBoostingRegressor(learning_rate=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(learning_rate=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(n_estimators=0)
            @test_throws ArgumentError GradientBoostingRegressor(subsample=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(subsample=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(validation_fraction=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(validation_fraction=1.0)
            @test_throws ArgumentError GradientBoostingRegressor(alpha=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(alpha=1.0)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=0.0)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=1.5)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_split=1)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0.0)
            # NovaML caps float min_samples_leaf at 0.5 (values > 0.5 make splits impossible)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0.6)
            @test_throws ArgumentError GradientBoostingRegressor(min_samples_leaf=0)
        end

        @testset "Fit and predict — squared_error" begin
            Random.seed!(42)
            X_train = randn(100, 3)
            y_train = 2.0 .* X_train[:, 1] .+ 0.5 .* X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.n_estimators_ == 50
            @test length(gbr.estimators_) == 50
            @test length(gbr.train_score_) == 50

            # Train loss decreases over boosting rounds
            @test gbr.train_score_[end] < gbr.train_score_[1]

            # Predictions are numeric vector
            preds = gbr(X_train)
            @test length(preds) == 100
            @test eltype(preds) <: AbstractFloat

            # Feature importances
            @test gbr.feature_importances_ !== nothing
            @test length(gbr.feature_importances_) == 3
            @test all(gbr.feature_importances_ .>= 0)
        end

        @testset "Fit and predict — absolute_error" begin
            Random.seed!(123)
            X_train = randn(80, 2)
            y_train = 3.0 .* X_train[:, 1] .+ randn(80) .* 0.2

            gbr = GradientBoostingRegressor(
                loss="absolute_error",
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                random_state=123
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.train_score_[end] < gbr.train_score_[1]

            preds = gbr(X_train)
            @test length(preds) == 80
            @test eltype(preds) <: AbstractFloat
        end

        @testset "Fit and predict — huber" begin
            Random.seed!(456)
            X_train = randn(80, 2)
            y_train = 1.5 .* X_train[:, 1] .- 0.5 .* X_train[:, 2] .+ randn(80) .* 0.3

            gbr = GradientBoostingRegressor(
                loss="huber",
                alpha=0.9,
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                random_state=456
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            @test gbr.train_score_[end] < gbr.train_score_[1]

            preds = gbr(X_train)
            @test length(preds) == 80
            @test eltype(preds) <: AbstractFloat
        end

        @testset "Early stopping" begin
            Random.seed!(789)
            X_train = randn(200, 2)
            y_train = X_train[:, 1] .+ randn(200) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.5,
                max_depth=3,
                n_iter_no_change=10,
                tol=1e-4,
                validation_fraction=0.2,
                random_state=789
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            # Should stop before reaching 500 estimators
            @test gbr.n_estimators_ < 500
        end

        @testset "Predict not fitted" begin
            gbr = GradientBoostingRegressor()
            @test_throws ErrorException gbr(randn(5, 2))
        end

        @testset "Subsample" begin
            Random.seed!(101)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                subsample=0.8,
                learning_rate=0.1,
                max_depth=3,
                random_state=101
            )
            gbr(X_train, y_train)

            @test gbr.fitted == true
            preds = gbr(X_train)
            @test length(preds) == 100
        end

        @testset "Warm start continuation" begin
            Random.seed!(202)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ X_train[:, 2] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=202
            )
            gbr(X_train, y_train)
            @test gbr.fitted == true
            @test length(gbr.estimators_) == 30

            # Continue training with more estimators
            gbr.n_estimators = 60
            gbr(X_train, y_train)
            @test length(gbr.estimators_) == 60
            @test gbr.n_estimators_ == 60
            @test length(gbr.train_score_) == 60
        end

        @testset "Warm start — lowering n_estimators errors" begin
            Random.seed!(303)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=303
            )
            gbr(X_train, y_train)
            @test length(gbr.estimators_) == 30

            # Lowering n_estimators with warm_start should throw
            gbr.n_estimators = 20
            @test_throws ArgumentError gbr(X_train, y_train)
        end

        @testset "Warm start — same n_estimators is no-op" begin
            Random.seed!(404)
            X_train = randn(100, 2)
            y_train = X_train[:, 1] .+ randn(100) .* 0.1

            gbr = GradientBoostingRegressor(
                n_estimators=30,
                learning_rate=0.1,
                max_depth=3,
                warm_start=true,
                random_state=404
            )
            gbr(X_train, y_train)
            preds1 = gbr(X_train)

            # Re-fit with same n_estimators — should keep same estimators
            gbr(X_train, y_train)
            preds2 = gbr(X_train)
            @test length(gbr.estimators_) == 30
            @test preds1 == preds2
        end
    end

end

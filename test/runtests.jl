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

end

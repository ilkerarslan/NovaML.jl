using Nova
using Test

using Nova.LinearModel: Perceptron

@testset "Nova.jl" begin
    @testset "Perceptron Tests" begin
        @testset "Constructor" begin
            p = Perceptron()
            @test p.η == 0.01
            @test p.num_iter == 100
            @test p.random_state === nothing
            @test p.optim_alg == "SGD"
            @test p.batch_size == 32
            @test p.fitted == false
            @test isempty(p.w)
            @test p.b ≤ 0.01 
            @test isempty(p.losses)
    
            p_custom = Perceptron(η=0.1, num_iter=200, random_state=42, optim_alg="MiniBatch", batch_size=64)
            @test p_custom.η == 0.1
            @test p_custom.num_iter == 200
            @test p_custom.random_state == 42
            @test p_custom.optim_alg == "MiniBatch"
            @test p_custom.batch_size == 64    
            
        end
    
        @testset "Prediction" begin
            p = Perceptron()
            p.w = [1.0, -1.0]
            p.b = 0.0
    
            @test p([2.0, 1.0]) == 1
            @test p([-1.0, 2.0]) == 0
    
            X = [2.0 1.0; -1.0 2.0; 3.0 -3.0]
            @test p(X) == [1, 0, 1]
        end
    
        @testset "Training" begin
            # Simple linearly separable dataset
            X = [1 2; 2 3; 3 1; 4 3]
            y = [0, 0, 1, 1]
    
            p = Perceptron(num_iter=100, random_state=42)
            p(X, y)
    
            @test p.fitted == true
            @test !isempty(p.w)            
            @test !isempty(p.losses)            
    
            # Check if the model can correctly classify the training data
            @test all(p(X) .== y)
        end
    
        @testset "Edge Cases" begin
            p = Perceptron()            
                      
            # Mismatched dimensions
            @test_throws DimensionMismatch p([1.0, 2.0, 3.0])  # Assuming weights are not initialized            
        end
    end
end

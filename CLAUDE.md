# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NovaML.jl is a Julia ML framework with a scikit-learn-style API, currently in alpha (version 0.4.x). Module/function names intentionally mirror scikit-learn, but NovaML is a native Julia implementation, not a wrapper.

## Common commands

Run from the repo root in a Julia REPL or via `julia --project=.`:

- Run the full test suite: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Tests live in `test/runtests.jl` (single file). To run one `@testset`, edit the file or use `Pkg.test(test_args=["..."])` — there is no per-file selector.
- Resolve/install deps: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Build docs locally: `julia --project=docs/ docs/make.jl` (uses Documenter; output lands in `docs/build/`).

CI runs the test suite on Julia 1.6, 1.10, and `pre` (Ubuntu only) via `.github/workflows/CI.yml`. The `compat` floor in `Project.toml` is `julia = "1.6.7"` — don't use syntax newer than 1.6 in `src/` without bumping it.

## Architecture: the functor convention

The single most important thing to internalize before editing models. Every estimator is a `mutable struct` that is also a callable, and overloads on argument shape decide whether it fits, predicts, or transforms. This is enforced by abstract types in `src/_types.jl`:

```julia
abstract type AbstractModel end
abstract type AbstractMultiClass <: AbstractModel end
abstract type AbstractScaler end

# Default matrix dispatch: predict one row at a time
(m::AbstractModel)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]
```

Convention for every supervised model `M <: AbstractModel`:

- `m(X::AbstractMatrix, y::AbstractVector)` — **fits** the model. Mutates `m` in place; sets `m.fitted = true`. Should also set learned-attribute fields (e.g. `m.w`, `m.b`, `m.losses`).
- `m(x::AbstractVector)` — predicts for a single sample. The default `(m::AbstractModel)(X::AbstractMatrix)` calls this row-by-row, so defining the scalar/vector overload is usually sufficient.
- `m(X::AbstractMatrix; type=...)` — for models with optional output modes (e.g. `LogisticRegression(X; type=:probs)` returns class probabilities instead of labels).
- Pre-fit prediction should throw with a message like *"This X instance is not fitted yet."* — see `LogisticRegression` for the exact pattern.

Convention for scalers/transformers `T <: AbstractScaler`:

- `t(X)` does fit-then-transform on first call, transform-only on subsequent calls. State is the `fitted::Bool` field. See `src/PreProcessing/StandardScaler.jl`.
- Optional `type=:transform` / `type=:inverse` keyword controls direction.

This functor convention is what makes pipelines work. `src/PipeLines/PipeLines.jl` is ~30 lines and just walks `steps`, calling each functor; the last step gets `(data, y)` in training mode and `(data; kwargs...)` in prediction mode. The `|>` examples in the README rely on the same shape.

Hyperparameters and learned parameters share the struct. Constructors take only keyword args with defaults and zero-initialize learned fields. `GridSearchCV` mutates these structs through `setproperty!` (see `_set_params!` in `src/_methods.jl`), so keep parameter names stable when renaming fields.

## Architecture: module layout

`src/NovaML.jl` is the top-level module and assembles submodules via `include`. Each submodule is its own directory under `src/` with a module-defining file of the same name (e.g. `src/LinearModel/LinearModel.jl`). Inside a submodule file, individual algorithms are pulled in via `include` and re-exported.

Submodules import shared infrastructure with relative paths: `import ...NovaML: AbstractModel, net_input, sigmoid`. The three dots are necessary because the algorithm file is nested two levels deep (`NovaML` → `LinearModel` → `LogisticRegression.jl`). When adding a new algorithm file, copy this import style — using bare `using NovaML` will fail during package load.

Shared math lives in `src/_methods.jl`: `net_input`, `sigmoid`, `softmax`, `default_score`, and the GridSearchCV/RandomSearchCV parameter-expansion helpers `_generate_param_combinations`, `_param_product`, `_set_params!`. Prefer extending these over duplicating math inside individual models.

## Adding a new algorithm — checklist

1. Create `src/<Submodule>/<Name>.jl`. Subtype `AbstractModel` (or `AbstractMultiClass` / `AbstractScaler`).
2. Use `import ...NovaML: AbstractModel, ...` for shared symbols.
3. Define struct fields in two groups by comment: learned params (incl. `fitted::Bool`) and hyperparameters. Keyword-only constructor.
4. Implement the functor methods following the convention above. Validate enum-like `Symbol` hyperparameters in the constructor with `throw(ArgumentError(...))` — see `LogisticRegression`'s solver check.
5. Add `include("<Name>.jl")` and an `export` line to the submodule's main file.
6. Add an `@testset` block to `test/runtests.jl` covering: constructor defaults, constructor validation, fit, predict-not-fitted error, and basic numeric correctness. The `KNeighborsRegressor` testset is the current reference shape.
7. Update `README.md` if the algorithm is user-facing.

## Notes on existing patterns worth preserving

- `LogisticRegression` supports multiple solvers (`:sgd`, `:batch`, `:minibatch`, `:lbfgs`, `:liblinear`) selected by symbol. New iterative linear models should follow the same pattern rather than inventing a new dispatch scheme.
- Loss/iteration history is stored on the struct as `m.losses::Vector{Float64}` and `empty!`'d at the start of each fit.
- Greek-letter field names are used freely (`η`, `λ`) — they are valid Julia identifiers and match the math notation in docs. Keep them.
- `default_score` in `src/_methods.jl` auto-switches between R² and accuracy based on the eltype of `y` — model-selection code uses it as the scoring fallback.

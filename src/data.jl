"""
Data structures and constructors for DoubleML.

Provides `DoubleMLData` for storing outcome, treatment, and covariates.
Default numeric type is Float32 for ML performance.
"""

"""
    dtype(data::DoubleMLData{T}) where {T} -> Type{T}

Return the numeric type used in the data container.

Can be called as a function `dtype(data)` or as a property `data.dtype`.
"""
dtype(::DoubleMLData{T}) where {T} = T

# Allow data.dtype as a property accessor for backward compatibility
function Base.getproperty(data::DoubleMLData, sym::Symbol)
    if sym === :dtype
        return dtype(data)
    else
        return getfield(data, sym)
    end
end

"""
    check_binary(v::AbstractVector) -> Bool

Check if a variable is binary with values 0 and 1.
"""
function check_binary(v::AbstractVector)
    non_missing = collect(skipmissing(v))
    isempty(non_missing) && return false
    return all(x -> x == 0 || x == 1, non_missing)
end

"""
    _validate_columns_exist(df, y_col, d_col, x_cols)

Validate that all specified columns exist in the DataFrame.
"""
function _validate_columns_exist(
        df::DataFrame, y_col::Symbol, d_col::Symbol, x_cols::Vector{Symbol}
    )
    cols = propertynames(df)

    !(y_col in cols) && throw(
        ArgumentError("Outcome column '$y_col' not found. Available: $(join(cols, ", "))")
    )

    !(d_col in cols) && throw(
        ArgumentError("Treatment column '$d_col' not found. Available: $(join(cols, ", "))")
    )

    for x_col in x_cols
        !(x_col in cols) && throw(
            ArgumentError("Covariate column '$x_col' not found. Available: $(join(cols, ", "))")
        )
    end

    return nothing
end

"""
    _validate_dimensions(y, d, x, y_col, d_col, x_cols)

Validate that all data components have consistent dimensions.
"""
function _validate_dimensions(
        y::AbstractVector,
        d::AbstractVector,
        x::AbstractMatrix,
        y_col::Symbol,
        d_col::Symbol,
        x_cols::Vector{Symbol}
    )
    n_y = length(y)
    n_d = length(d)
    n_x = size(x, 1)

    n_y != n_d && throw(
        DimensionMismatch(
            "Outcome ($y_col) has $n_y observations, " *
                "but treatment ($d_col) has $n_d observations"
        )
    )

    n_y != n_x && throw(
        DimensionMismatch(
            "Outcome ($y_col) has $n_y observations, " *
                "but covariates have $n_x observations"
        )
    )

    return n_y, size(x, 2)
end

"""
    _validate_numeric_type(::Type{T}, context::String) where T

Validate that a type is numeric (AbstractFloat).
"""
function _validate_numeric_type(::Type{T}, context::String) where {T}
    !(T <: AbstractFloat) && throw(ArgumentError("$context must be floating point type. Got: $T"))
    return true
end

"""
    _extract_covariates(df, x_cols, ::Type{T}) where T

Extract covariates as a properly typed matrix.
"""
function _extract_covariates(df::DataFrame, x_cols::Vector{Symbol}, ::Type{T}) where {T}
    first_col = df[!, first(x_cols)]
    n = length(first_col)
    k = length(x_cols)
    x = Matrix{T}(undef, n, k)

    for (j, col) in enumerate(x_cols)
        col_data = df[!, col]
        col_type = eltype(col_data)

        !(col_type <: AbstractFloat) && throw(
            ArgumentError("Covariate column '$col' must be floating point type. Got: $col_type")
        )

        x[:, j] = T.(col_data)
    end

    return x
end

"""
    DoubleMLData(df::DataFrame, ::Type{T}; y_col, d_col, x_cols) where T<:AbstractFloat

Create DoubleMLData with explicit numeric type.

# Arguments
- `df::DataFrame`: Source data frame
- `T::Type`: Numeric type for storage (Float32 or Float64)
- `y_col::Symbol`: Column name for outcome variable
- `d_col::Symbol`: Column name for treatment variable
- `x_cols::Vector{Symbol}`: Column names for covariates

# Returns
- `DoubleMLData{T}`: Data container

# Examples
```julia
# Float32 (default, ML-efficient)
data = DoubleMLData(df, Float32; y_col=:y, d_col=:d, x_cols=[:x1, :x2])

# Float64 (higher precision)
data = DoubleMLData(df, Float64; y_col=:y, d_col=:d, x_cols=[:x1, :x2])
```
"""
function DoubleMLData(
        df::DataFrame, ::Type{T};
        y_col::Symbol,
        d_col::Symbol,
        x_cols::Vector{Symbol}
    ) where {T <: AbstractFloat}

    _validate_columns_exist(df, y_col, d_col, x_cols)

    y_raw = df[!, y_col]
    _validate_numeric_type(eltype(y_raw), "Outcome ($y_col)")
    y = T.(y_raw)

    x = _extract_covariates(df, x_cols, T)

    d = df[!, d_col]

    n_obs, dim_x = _validate_dimensions(y, d, x, y_col, d_col, x_cols)

    return DoubleMLData{T, typeof(d)}(y, d, x, n_obs, dim_x, y_col, d_col, x_cols)
end

"""
    DoubleMLData(df::DataFrame; y_col, d_col, x_cols)

Create DoubleMLData with Float32 as default numeric type.
"""
function DoubleMLData(
        df::DataFrame;
        y_col::Symbol,
        d_col::Symbol,
        x_cols::Vector{Symbol}
    )
    return DoubleMLData(df, Float32; y_col = y_col, d_col = d_col, x_cols = x_cols)
end

"""
    DoubleMLData(df::DataFrame, y_col::Symbol, d_col::Symbol)

Shorthand constructor that infers covariates automatically.

All columns except `y_col` and `d_col` are used as covariates.
Defaults to Float32.
"""
function DoubleMLData(df::DataFrame, y_col::Symbol, d_col::Symbol)
    cols = propertynames(df)

    !(y_col in cols) && throw(
        ArgumentError(
            "Outcome column '$y_col' not found in DataFrame. " *
                "Available: $(join(cols, ", "))"
        )
    )

    !(d_col in cols) && throw(
        ArgumentError(
            "Treatment column '$d_col' not found in DataFrame. " *
                "Available: $(join(cols, ", "))"
        )
    )

    x_cols = filter(c -> c != y_col && c != d_col, cols)

    isempty(x_cols) && throw(
        ArgumentError(
            "No covariate columns found. DataFrame must contain columns " *
                "other than '$y_col' and '$d_col'."
        )
    )

    return DoubleMLData(df, Float32; y_col = y_col, d_col = d_col, x_cols = x_cols)
end

"""
    DoubleMLData(df::DataFrame, y_col::Symbol, d_col::Symbol, ::Type{T})

Shorthand constructor with explicit numeric type.
"""
function DoubleMLData(df::DataFrame, y_col::Symbol, d_col::Symbol, ::Type{T}) where {T}
    cols = propertynames(df)

    !(y_col in cols) && throw(
        ArgumentError(
            "Outcome column '$y_col' not found in DataFrame. " *
                "Available: $(join(cols, ", "))"
        )
    )

    !(d_col in cols) && throw(
        ArgumentError(
            "Treatment column '$d_col' not found in DataFrame. " *
                "Available: $(join(cols, ", "))"
        )
    )

    x_cols = filter(c -> c != y_col && c != d_col, cols)

    isempty(x_cols) && throw(
        ArgumentError(
            "No covariate columns found. DataFrame must contain columns " *
                "other than '$y_col' and '$d_col'."
        )
    )

    return DoubleMLData(df, T; y_col = y_col, d_col = d_col, x_cols = x_cols)
end

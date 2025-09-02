module Utils 
# Exporing functions 
export dist_euclid, exp_sq_kernel, M_g
# Imports
using LinearAlgebra: I

# Eucledian Distance
@doc """
Computes the eucledian distance between regions, this funvtion
is used to find the distance between points.
Inputs
    - x : lat/lon values
    - z : lat/lon values
Note for this problem x and z are the same as we are trying to find
the distance between each point and all other points.
"""->
function dist_euclid(x::Array{Float64},z::Array{Float64})::Array{Float64}
    # If x or z is in the shape (n_grid_pts,) -> (n_grid_pts, 1)
    if ndims(x) == 1
        x = reshape(x, (length(x), 1)) # if (n_grd_pts,) -> (n_grd_pts, 1)
    end
    if ndims(z) == 1
        z = reshape(z, (length(z), 1))
    end
    # number of grid pints and lat/lon dims
    n_x, m = size(x) # n_grd_pts , 2
    n_z, m_z = size(z)
    @assert m == m_z
    # contruct a matrix wuth shape (n_grd_pts, n_grd_pts)
    delta = zeros((n_x,n_z)) #(n_grd_pts, n_grd_pts)
    for d = 1:m
        # in the first iter they consider lat in 2nd ite they consider lon
        x_d = x[:,d] #(2618,)
        z_d = z[:,d] #(2618,)
        # compute distance between each point and all other points
        delta .+= (x_d .- reshape(z_d, (1, length(z_d)))).^2
    end
    return sqrt.(delta) #(2618,2618)
end

# Exponential Square Kernel
@doc """
Exponential Square Kernel
Inputs
    - x : Lat/Lon for Grid
    - z : Lat/Lon for Grid
    For this problem x and z are the same 
"""->
function exp_sq_kernel(x::Matrix{Float64},z::Matrix{Float64},k_var::Float64,k_length::Float64,noise::Float64,jitter::Float64=1.0e-4)::Matrix{Float64}
    dist = dist_euclid(x,z) #(n_grd_pts,n_grid_pts)
    deltaXsq = (dist ./ k_length).^2 #(n_grd_pts, n_grd_pts)
    k = k_var .* exp.(-0.5 .* deltaXsq) #(n_grd_pts, n_grd_pts)
    k .+= (noise + jitter) .* Matrix(I, (size(x)[1],size(x)[1])) #(n_grd_pts, n_grd_pts)
    return k #(n_grd_pts, n_grd_pts)
end

# Aggregate points in region func 
@doc"""
Used to aggregated values per region 
Inputs 
    - M : Matrix with binary entries $m_{ij}, $ showing whether point $j$ is in polygon $i$
        - shape : (n_regions, n_grd_pts) ; e.g : (9, 2618)
        - This is the variable pol_pt_lo or pol_pt_hi
    - g : Is a vector of GP draws over the grid
        - shape : (n_grd_pts,) ; e.g (2618,)
        - This is the gp function
    - matmul(M,g) gives a vector sum over each polygon
"""->
M_g(M,g) = M * g #(n_regions,) e.g (9,)

# Main - Just to test the Utils module
if abspath(PROGRAM_FILE) == @__FILE__
    using .Utils
    x = rand(2618, 2) # random lat/lon values
    @show size(x)

    dist = Utils.dist_euclid(x,x)
    @show size(dist)

    k = Utils.exp_sq_kernel(x, x, 1.0, 1.0, 1e-4)
    @show size(k)

    pol_pt_lo = Int.(rand(Bool, 9, 2618)) # random binary matrix for low polygon points
    f = rand(2618) # random GP function values
    aggregated_lo = Utils.M_g(pol_pt_lo, f)
    @show size(aggregated_lo)
    
end
end


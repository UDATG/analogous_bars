module gen_points

using Clustering
using Distances
using Distributions
export generate_points,
       get_centroids

function generate_points()
    #1. Define a large circle
    #2. Sample 'center' points from the large circle
    #3. For each 'center' points, sample a disk around it
    
    # generate 'center' points
    n_center = 20
    theta = rand(Uniform(0,2*pi), n_center)
    distance = rand(Normal(2, 0.2), n_center);

    center = zeros(n_center,2)
    for i=1:n_center
        center[i, 1] = distance[i] * cos(theta[i]) 
        center[i, 2] = distance[i] * sin(theta[i])
    end

    # define the covariance matrix for 2D Gaussian
    sigma = Array{Float64}(undef, 2, 2)
    sigma[1,1] = 0.05
    sigma[1, 2] = 0
    sigma[2, 1] = 0
    sigma[2, 2] = 0.05

    # sample points around centers
    n_samples = 10
    points = Matrix{Float64}(undef, 2, 0)

    for i = 1:n_center
        sampled_points = rand(MvNormal(center[i,:], sigma), n_samples)
        points = hcat(points, sampled_points)
    end
    return points
    
end

function get_centroids(points, n_clusters)
    
    # compute distance matrix
    D = pairwise(Euclidean(), points, points, dims=2);
    
    # clustering
    clustering = hclust(D, linkage = :single)
    clusters = cutree(clustering, k = n_clusters)
    
    # get centroids of each cluster
    centroids = Matrix{Float64}(undef, 2,0)
    for i =1:n_clusters
        index = findall(x -> x == i, clusters)
        c = mean(points[:, index], dims = 2)
        centroids = hcat(centroids, c)
    end
    
    return clusters, centroids
end

function compute_distances(points, centroids)
    # total points
    P = hcat(points, centroids)
    n_points = size(points)[2]
    n_centroids = size(centroids)[2]

    # compute distances
    D = pairwise(Euclidean(), P, P, dims=2)

    # Define submatrices 
    D_R1 = D[1:n_points, 1:n_points]
    D_R2 = D[n_points+1:end, n_points+1:end]
    D_R1_R2 = D[1:n_points, n_points+1:end]
        # rows (landmarks): R1
        # columns (witness) : R2
    D_R2_R1 = D[n_points+1:end, 1:n_points];
        # rows (landmarks): R2
        # columns (witness) : R1
    return D
end


end
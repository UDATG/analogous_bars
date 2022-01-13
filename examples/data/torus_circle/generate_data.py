# sample 300 points from torus S1 x S1 and 100 points from circle S1 (belonging to the first S1)
# Compute distance among the points

import math
import numpy as np
import h5py

def angle_distance(phi, theta):
    s = min(phi, theta)
    l = max(phi, theta)
    distance = min(l-s, s + 2 * math.pi - l)

    return distance

def compute_distances(X):
    n_points = X.shape[0]
    d = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i+1, n_points):
            d_phi = angle_distance(X[i,0], X[j,0])
            d_theta = angle_distance(X[i,1], X[j,1])
            distance = math.sqrt(d_phi**2 + d_theta **2)
            d[i][j] = distance
            d[j][i] = distance        
    return d

def main():
    n_torus = 300
    n_circle = 100
    # generate 300 points on S1 x S1 
    torus = np.random.uniform(0, 2*math.pi, (300, 2))
    # columns 0, 1, 2: phi, psi, theta

    # generate 100 points from S1
    circle = np.zeros((20, 2))
    for i in range(20):
        circle[i,0] = np.random.uniform(0, 2*math.pi)

        # add some noise
        circle[i,1] = np.random.normal(math.pi, 0.2)

    data = np.concatenate((torus, circle), axis=0)
    distance = compute_distances(data)
    
    # break into 4 distance matrices
    # Define submatrices 
    D_torus = distance[:n_torus, :n_torus]
    D_circle = distance[n_torus:, n_torus:]
    D_torus_circle = distance[:n_torus, n_torus:]
        # rows (landmarks): torus
        # columns (witness) : circle
    D_circle_torus = distance[n_torus:, :n_torus];
        # rows (landmarks): circle
        # columns (witness) : torus
    np.savetxt("distance.csv", distance)
    np.savetxt("distance_torus.csv", D_torus)
    np.savetxt("distance_circle.csv", D_circle)
    np.savetxt("distance_torus_circle.csv", D_torus_circle)
    np.savetxt("distance_circle_torus.csv", D_circle_torus)

    # save coordinates for visualization comparison
    hf = h5py.File("coords.h5", "w")
    hf.create_dataset("torus", data = np.transpose(torus))
    hf.create_dataset("circle", data = np.transpose(circle))
    hf.close()
if __name__ == '__main__':
    main()


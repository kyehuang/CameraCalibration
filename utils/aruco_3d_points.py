import numpy as np

# Define 3D points for each Aruco ID
aruco_3d_points_dict = {
    # Example: Aruco ID 0 is a square with side length 1 unit
    42: np.array([
        [-0.5, 0, 0.5],   # topLeft in 3D space
        [0.5,  0, 0.5],    # topRight in 3D space
        [0.5,  0,-0.5],     # bottomRight in 3D space
        [-0.5, 0,-0.5]     # bottomLeft in 3D space
    ]),
    0: np.array([
        [2.5, 0, 1.5],   # topLeft in 3D space
        [3.5, 0, 1.5],
        [3.5, 0, 0.5],
        [2.5, 0, 0.5]
    ]),
    2: np.array([
        [6, -1.5, 0.5],   # topLeft in 3D space
        [6, -1.5, -0.5],
        [6, -0.5, -0.5],
        [6, -0.5, 0.5]
    ]),
    # Add more Aruco IDs and their corresponding 3D points as needed
}
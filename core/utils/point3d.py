import numpy as np


def is_nan(val):
    return val is None or np.isnan(val)


def project_world3d_to_2d(keypoints_3d, K, R, T):
    # ref from https://hackmd.io/@-uoCoKkVTp2zUnRuNFpEBg/Python-OpenCV%E7%AD%86%E8%A8%982#Relationship-of-2D-x-Intrinsic-x-Extrinsic-x-3D
    
    # keypoints_3d: numpy array of shape (Keypoints, 3)
    # K: camera intrinsic matrix (3x3)
    # R: rotation matrix (3x3)
    # T: translation vector (3x1)
    
    # Homogeneous coordinates
    keypoints_3d_h = np.hstack((keypoints_3d, np.ones((keypoints_3d.shape[0], 1))))

    # Apply rotation and translation
    extrinsic_matrix = np.hstack((R, T))
    keypoints_3d_transformed = np.dot(extrinsic_matrix, keypoints_3d_h.T)

    # Project to 2D
    keypoints_2d_h = np.dot(K, keypoints_3d_transformed)

    # Convert back to inhomogeneous coordinates
    keypoints_2d = keypoints_2d_h[:2, :] / keypoints_2d_h[2, :]

    return keypoints_2d.T


def project_camera3d_to_2d(keypoints_3d, K):
    # keypoints_3d: numpy array of shape (Keypoints, 3)
    # K: camera intrinsic matrix (3x3)
    
    # Project to 2D
    keypoints_2d_h = np.dot(K, keypoints_3d.T)

    # Convert back to inhomogeneous coordinates
    keypoints_2d = keypoints_2d_h[:2, :] / keypoints_2d_h[2, :]
    
    return keypoints_2d.T


def transform_keypoints_to_novelview(keypoints_3d, R, T, R_prime, T_prime):
    """
    Transforms camera 3D keypoints from view0 coordinates to view1 coordinates.

    Args:
        keypoints_3d (np.array): 3D keypoints in view0 coordinates.
        R (np.array): Rotation matrix of view0.
        T (np.array): Translation vector of view0.
        R_prime (np.array): Rotation matrix of view1.
        T_prime (np.array): Translation vector of view1.
    
    Returns:
        np.array: 3D keypoints in view1 coordinates.
    """
    # Mark rows with [None, None, None] in keypoints_3d
    none_rows = np.all(keypoints_3d == None, axis=1)
    # Temporarily remove none_rows from keypoints_3d
    keypoints_3d_temp = keypoints_3d[~none_rows]

    if R is None:
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if T is None:
        T = np.array([[0], [0], [0]])

    keypoints_world = np.dot(np.linalg.inv(R), (keypoints_3d_temp.T - T))
    keypoints_prime = np.dot(R_prime, keypoints_world) + T_prime

    keypoints_prime = keypoints_prime.T
    # Insert the none_rows back into the return_2d array
    return_3d_final = np.empty((keypoints_3d.shape[0], 3))
    return_3d_final[none_rows] = None
    return_3d_final[~none_rows] = keypoints_prime
    return return_3d_final


def lift_2d_keypoints_to_camera3d(keypoints_2d, K):
    """
    Lift 2D keypoints to camera 3D keypoints coords given camera parameters K.
    keypoints_2d: np.array, [18 x 2]
    Returns: np.array, [18 x 3], 3D keypoints
    """
    # Mark rows with [None, None, None] in keypoints_3d
    none_rows = np.all(keypoints_2d == None, axis=1)
    # Temporarily remove none_rows from keypoints_3d
    keypoints_2d_temp = keypoints_2d[~none_rows]
    
    num_keypoints = keypoints_2d_temp.shape[0]
    keypoints_2d_homogeneous = np.hstack((keypoints_2d_temp, np.ones((num_keypoints, 1))))
    keypoints_3d = np.linalg.inv(K) @ keypoints_2d_homogeneous.T
    keypoints_3d = keypoints_3d.T
    
    # Insert the none_rows back into the return_3d array
    return_3d_final = np.empty((keypoints_2d.shape[0], 3))
    return_3d_final[none_rows] = None
    return_3d_final[~none_rows] = keypoints_3d
    return return_3d_final
    

def lift_2d_keypoints_to_world3d(keypoints_2d, K, R, T):
    """
    Lift 2D keypoints to 3D keypoints given camera parameters K, R, and T.
    keypoints_2d: np.array, [18 x 2]
    K: np.array, [3 x 3], camera intrinsic matrix
    R: np.array, [3 x 3], rotation matrix
    T: np.array, [3 x 1], translation vector
    Returns: np.array, [18 x 3], 3D keypoints
    """
    # Mark rows with [None, None, None] in keypoints_3d
    none_rows = np.all(keypoints_2d == None, axis=1)
    # Temporarily remove none_rows from keypoints_3d
    keypoints_2d_temp = keypoints_2d[~none_rows]
    
    num_keypoints = keypoints_2d_temp.shape[0]
    keypoints_2d_homogeneous = np.hstack((keypoints_2d_temp, np.ones((num_keypoints, 1))))
    keypoints_3d = np.linalg.inv(K) @ keypoints_2d_homogeneous.T
    keypoints_3d = np.linalg.inv(R) @ keypoints_3d - T
    keypoints_3d = keypoints_3d.T
    
    # Insert the none_rows back into the return_3d array
    return_3d_final = np.empty((keypoints_2d.shape[0], 3))
    return_3d_final[none_rows] = None
    return_3d_final[~none_rows] = keypoints_3d
    return return_3d_final

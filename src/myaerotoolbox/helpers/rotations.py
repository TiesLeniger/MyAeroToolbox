import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_rotation(rotation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies a rotation defined by 'rotation_matrix' to a given array of points 
    """
    if points.ndim == 4:                                                                        # Apply the rotation
        rotated_points = np.einsum("ij,kmnj->kmni", rotation_matrix, points)
    elif points.ndim == 3:
        rotated_points = np.einsum("ij,mnj->mni", rotation_matrix, points)
    elif points.ndim == 2:
        rotated_points = np.einsum("ij,nj->ni", rotation_matrix, points)
    elif points.ndim == 1:
        rotated_points = rotation_matrix @ points
    else:
        logger.error(f"The dimension of the input array 'points' is too high: {points.ndim}, max supported: 4")
        raise NotImplementedError("Rotation for 'points' arrays with dimensions higher than 4 have not been implemented")
    
    return rotated_points

def rotate_points_about_x_axis(points: np.ndarray, angle: float, unit = "rad"):
    """
    Rotates points in 3D space (x, y, z in [m]) about the x-axis with 'angle'. 
    Angle is converted to rad if unit is specified to be "deg"
    """
    if unit == "deg":
        angle = np.deg2rad(angle)
    
    cos = np.cos(angle)
    sin = np.sin(angle)

    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cos,-sin],
                                [0.0, sin, cos]])
    
    rotated_points = apply_rotation(rotation_matrix, points)

    return rotated_points

def rotate_points_about_y_axis(points: np.ndarray, angle: float, unit = "rad"):
    """
    Rotates points in 3D space (x, y, z in [m]) about the y-axis with 'angle'. 
    Angle is converted to rad if unit is specified to be "deg"
    """
    if unit == "deg":
        angle = np.deg2rad(angle)
    
    cos = np.cos(angle)
    sin = np.sin(angle)

    rotation_matrix = np.array([[cos,  0.0, sin],
                                [0.0,  1.0, 0.0],
                                [-sin, 0.0, cos]])
    
    rotated_points = apply_rotation(rotation_matrix, points)

    return rotated_points

def rotate_points_about_z_axis(points: np.ndarray, angle: float, unit = "rad"):
    """
    Rotates points in 3D space (x, y, z in [m]) about the z-axis with 'angle'. 
    Angle is converted to rad if unit is specified to be "deg"
    """
    if unit == "deg":
        angle = np.deg2rad(angle)
    
    cos = np.cos(angle)
    sin = np.sin(angle)

    rotation_matrix = np.array([[cos,-sin, 0.0],
                                [sin, cos, 0.0],
                                [0.0, 0.0, 1.0]])
    
    rotated_points = apply_rotation(rotation_matrix, points)

    return rotated_points

def rotate_points_about_arbitrary_axis(axis_start: np.ndarray, axis_end: np.ndarray, points: np.ndarray, angle: float, unit: str = "rad"):
    """
    Rotates points in 3D space (x, y, z in [m]) about an arbitrary axis through axis_start and axis_end
    """
    if unit == "deg":
        angle = np.deg2rad(angle)
    
    cos = np.cos(angle)
    sin = np.sin(angle)
    
    k = axis_end - axis_start                                                               # Axis vector k
    norm_k = np.linalg.norm(k)                                                              # The norm of the axis vector
    if norm_k < 1e-6:
        logger.error(f"Axis start point and end point are too close together")
        raise ValueError(f"Axis start point and end point are too close together")
    else:
        k = k / norm_k                                                                      # Normalize the axis vector

    R1 = np.eye(3, 3)                                                                           # 3 by 3 Identity matrix
    R2 = k[:, np.newaxis] * k.transpose()[np.newaxis, :]                                        # Outer product of the axis vector
    R3 = np.array([[0.0, -k[2], k[1]],                                                          # Skew symmetric matrix of k
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]])
    
    rotation_matrix = cos*R1 + (1-cos)*R2 + sin*R3                                              # Construct the rotation matrix with R1, R2 and R3
   
    rotated_points = apply_rotation(rotation_matrix, points)

    return rotated_points
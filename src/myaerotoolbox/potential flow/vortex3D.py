import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)

def v_ind_by_semi_inf_vortex(P: np.ndarray, A: np.ndarray, r0: np.ndarray, gamma: Union[int, float, np.ndarray], cross_tolerance: float = 1e-6) -> np.ndarray:
    """
    Calculate the induced velocity at point(s) P by (a) semi-infinite vortex filament(s) starting in A with circulation gamma

    Parameters
    ----------
    P : np.ndarray
        Coordinates of point(s) where the induced velocity is calculated, last dimension must have 3 components.
        If P has a higher dimension than 2, it will be reshaped to P.reshape(-1, 3)
    A : np.ndarray
        Coordinates of the starting point of the vortex filament, last dimension must have 3 components.
        If A has a higher dimension than 2, it will be reshaped to A.reshape(-1, 3)
    r0 : np.ndarray
        Direction vector of the vortex filaments, last dimension must have 3 components.
    gamma : int, float, np.ndarray
        The circulation values of the vortex filaments. If gamma is a scalar, the same circulation is applied to all vortex filaments.
    cross_tolerance : float, optional
        Tolerance to apply to the cross product between r0 and AP to avoid singularities, by default 1e-6
    """
    if P.shape[-1] != 3 or A.shape[-1] != 3 or r0.shape[-1] != 3:
        logger.error("Input vectors must have 3 components")
        raise ValueError("Input vectors must have 3 components")
    if P.ndim > 2:
        logger.debug(f"P has more than 2 dimensions ({P.shape}), reshaping to 2D array")
        P = P.reshape(-1, 3)                                    # Reshape P to a 2D array, shape (N, 3)
        P = P[:, np.newaxis, :]                                 # P now has shape (N, 1, 3)  
    elif P.ndim == 1:
        P = P[np.newaxis, np.newaxis, :]                        # P now has shape (1, 1, 3)        
    if A.ndim > 2:
        logger.debug(f"A has more than 2 dimensions ({A.shape}), reshaping to 2D array")
        A = A.reshape(-1, 3)                                    # Reshape A to a 2D array, shape (M, 3)     
        A = A[np.newaxis, :, :]                                 # A now has shape (1, M, 3)
    elif A.ndim == 1:
        A = A[np.newaxis, np.newaxis, :]                        # A now has shape (1, 1, 3)
    
    r0norm = r0 / np.linalg.norm(r0)                            # Normalized direction vector of the vortex filament, shape (3,)
    AP = P - A                                                  # Vectors from point A to point P, shape (N, M, 3)
    normAP = np.linalg.norm(AP, axis=-1, keepdims=True)         # Length of vector AP

    r0normxAP = np.cross(r0norm, AP, axis=-1)                   # Cross product between r0norm and AP, shape (N, M, 3)
    
    invalid_indices = np.linalg.norm(r0normxAP, axis=-1) < cross_tolerance  # Find indices where the cross product is zero

    r0normxAP[invalid_indices] = 0                              # Set the cross product to zero where it is invalid
    normAP[invalid_indices] = 1                                 # Set the length of AP to 1 where it is invalid

    v_ind = np.zeros((P.shape[0], A.shape[1], 3))               # Initialize the induced velocity array, shape (N, M, 3)

    v_ind = r0normxAP/ (normAP*(normAP - np.einsum("k,ijk->ij", r0norm, AP)[:, :, np.newaxis]))     # Calculate the induced velocity, shape (N, M, 3)
    
    if isinstance(gamma, np.ndarray):
        if gamma.ndim > 1:
            gamma = gamma.flatten()
        if gamma.shape[0] != A.shape[1]:
            logger.error("Length of gamma must be equal to the number of vortex filaments")
            raise ValueError("Length of gamma must be equal to the number of vortex filaments")
        v_ind = v_ind * gamma[np.newaxis, :, np.newaxis] * (1/(4*np.pi))
    elif isinstance(gamma, (int, float)):
        v_ind = v_ind * gamma * (1/(4*np.pi))
    else:
        logger.error("gamma must be a scalar or a numpy array")
        raise ValueError("gamma must be a scalar or a numpy array")

    return v_ind

def v_ind_by_finite_vortex(P: np.ndarray, A: np.ndarray, B: np.ndarray, gamma: Union[int, float, np.ndarray], cross_tolerance: float = 1e-6) -> np.ndarray:
    """
    Compute the velocity induced at observation points by a finite vortex filament using the Biot-Savart law.

    This vectorized function calculates the induced velocity at multiple observation points (P) due to a set of
    vortex filaments defined by their start (A) and end (B) points. The vortex filaments are assigned a circulation
    strength (`gamma`), which can be scalar or a vector corresponding to each filament.

    Parameters
    ----------
    P : np.ndarray
        Array of observation points with shape (N, 3) or equivalent. Each row represents a 3D point where
        the induced velocity is computed.
    A : np.ndarray
        Array of vortex filament start points with shape (M, 3) or equivalent. Each row represents a 3D point.
    B : np.ndarray
        Array of vortex filament end points with shape (M, 3) or equivalent. Each row represents a 3D point.
    gamma : Union[int, float, np.ndarray]
        Circulation strength of the vortex filaments. If scalar, the same value is applied to all filaments.
        If array, it must have shape (M,) matching the number of vortex filaments.
    cross_tolerance : float, optional
        Tolerance for determining if observation points are on the vortex line. Defaults to `1e-6`.

    Returns
    -------
    np.ndarray
        Induced velocity at observation points, with shape (N, M, 3). The velocity for each observation point
        is computed for all vortex filaments.

    Raises
    ------
    ValueError
        If input vectors do not have the required shape in the last axis.
        If the shape of A and B do not match
        If `gamma` is an array and does not match the number of filaments.
    
    Notes
    -----
    - The Biot-Savart law is used to compute the induced velocity:
      v = (gamma / (4 * pi)) * ((r1 x r2) / |r1 x r2|^2) * (r1/|r1| - r2/|r2|)
      where r1 = P - A and r2 = P - B.
    - Points on the vortex line (within `cross_tolerance`) are treated as having zero induced velocity
      to avoid singularities.
    - Input arrays with higher dimensions are reshaped to (N, 3) for observation points and (M, 3) for vortex filaments.
    """

    if P.shape[-1] != 3 or A.shape[-1] != 3 or B.shape[-1] != 3:
        logger.error("Input vectors must have 3 components")                                    # Validate the input vectors 
        raise ValueError("Input vectors must have 3 components")
    elif A.shape != B.shape:                                                                    # Validate the start and end points of the vortex filaments
        logger.error("A (startpoints) has a different shape than B (endpoints)")
        raise ValueError("A (startpoints) has a different shape than B (endpoints)")
    if P.ndim > 2:
        logger.debug(f"P has more than 2 dimensions ({P.shape}), reshaping to 2D array")
        P = P.reshape(-1, 3)                                                                    # Reshape P to a 2D array, shape (N, 3)
        P = P[:, np.newaxis, :]                                                                 # shape (N, 1, 3)  
    elif P.ndim == 1:
        P = P[np.newaxis, np.newaxis, :]                                                        # shape (1, 1, 3)        
    if A.ndim > 2:
        logger.debug(f"A has more than 2 dimensions ({A.shape}), reshaping to 2D array")
        A = A.reshape(-1, 3)                                                                    # Reshape A to a 2D array, shape (M, 3)     
        A = A[np.newaxis, :, :]                                                                 # shape (1, M, 3)
    elif A.ndim == 1:   
        A = A[np.newaxis, np.newaxis, :]                                                        # A now has shape (1, 1, 3), M = 1
    if B.ndim > 2:
        logger.debug(f"A has more than 2 dimensions ({B.shape}), reshaping to 2D array")
        B = B.reshape(-1, 3)                                                                    # Reshape A to a 2D array, shape (M, 3)     
        B = B[np.newaxis, :, :]                                                                 # shape (1, M, 3)
    elif A.ndim == 1:   
        B = B[np.newaxis, np.newaxis, :]                                                        # A now has shape (1, 1, 3), M = 1

    AB = np.squeeze((B - A), axis = 0)                                                          # Vectors from A to B, shape (M, 3)
    AP = P - A                                                                                  # Vectors from A to P, shape (N, M, 3)
    normAP = np.linalg.norm(AP, axis = 2, keepdims = True)                                      # Norm of vectors in AP, shape (N, M, 1) with keepdims
    BP = P - B                                                                                  # Vectors from B to P, shape (N, M, 3)
    normBP = np.linalg.norm(BP, axis = 2, keepdims = True)                                      # Norm of vectors in BP, shape (N, M, 1) with keepdims

    APxBP = np.cross(AP, BP, axis = 2)                                                          # Cross product of vectors in AP and BP, shape (N, M, 3)
    normAPxBP = np.linalg.norm(APxBP, axis = 2)                                                 # Get the norm of all the vectors in the cross product, shape (N, M)

    invalid = normAPxBP < cross_tolerance                                                       # Where the cross product of AP and BP is 0, P is on the line between A and B

    APxBP[invalid] = 0                                                                          # Set invalid indices to 0
    normAPxBP[invalid] = 1                                                                      # Avoid division by zero error later on

    if not np.all(invalid):
        v_ind = APxBP / np.square(normAPxBP)                                                    # Shape (N, M, 3)
        v_ind *= np.einsum('jk,ijk->ij', AB, ((AP/normAP)-(BP/normBP)))[:, :, np.newaxis]       # Shape (N, M, 3) (einsum term expanded to (N, M, 1) for broadcasting)
        if isinstance(gamma, (int, float)):
            v_ind *= gamma/(4*np.pi)                                                            # Shape (N, M, 3)
        elif isinstance(gamma, np.ndarray):
            if gamma.ndim > 1:                                                                  # Ensure the shape of gamma is (M,)
                gamma = gamma.flatten()
            if gamma.shape[0] != A.shape[1]:
                logger.error("Gamma must have shape (M,) to match the number of vortex filaments")         
                raise ValueError("Gamma must have shape (M,) to match the number of vortex filaments")
            else:
                v_ind *= gamma[np.newaxis, np.newaxis, :]/(4*np.pi)                             # Give gamma new axes to allow for broadcasting
        else:
            logger.error("Gamma must be specified as a scalar or as a numpy array")
            raise ValueError("Gamma must be specified as a scalar or as a numpy array")
    else:
        logger.warning("The points in P are all on the filament lines between the points in A and B")
        v_ind =  np.zeros((P.shape[0], A.shape[1], 3))
    
    return v_ind

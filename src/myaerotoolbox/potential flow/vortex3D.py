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

import numpy as np
from typing import Union
from ..helpers import rotate_points_about_x_axis, rotate_points_about_y_axis, rotate_points_about_z_axis, rotate_points_about_arbitrary_axis
import logging

logger = logging.getLogger(__name__)
class WingSection:

    def __init__(self, num_chordwise: int, num_spanwise: int, airfoils: Union[list, tuple],
                 rootLE: np.ndarray, rootchord: float, taper_ratio: float, span: float, 
                 sweep_angle: float, sweep_loc: float, twist_angle: float, twist_loc: float, 
                 dihedral: float, angle_units: str = "deg"
                 ):
        if num_chordwise < 1:
            logger.error(f"Specify at least 1 chordwise panel")
            raise ValueError(f"Specify at least 1 chordwise panel")
        elif num_spanwise < 1:
            logger.error(f"Specify at least 1 spanwise panel")
            raise ValueError(f"Specify at least 1 spanwise panel")
        if rootLE.ndim != 1 or rootLE.shape != 3:
            logger.error(f"rootLE must be given as a 1D numpy array of shape (3,), got {rootLE.shape}")
            raise ValueError(f"rootLE must be given as a 1D numpy array of shape (3,), got {rootLE.shape}")
        else:
            self.rootLE = rootLE                                # Leading edge point of the root of the section (geometry is built up w.r.t this point)
        if rootchord < 0:
            logger.error(f"The root chord can not be negative, got {rootchord}")
            raise ValueError(f"The root chord can not be negative, got {rootchord}")
        else:
            self.croot = rootchord                              # Root chord of the section
        if not (0 < taper_ratio <= 1):
            logger.error(f"The value for the taper ratio should be 0 < taper <= 1, got {taper_ratio}")
            raise ValueError(f"The value for the taper ratio should be 0 < taper <= 1, got {taper_ratio}")
        else:
            self.taper = taper_ratio                            # Taper ratio of the section (tip chord/root chord)
        self.ctip = rootchord * taper_ratio                     # Tip chord of the section
        if span <= 0:
            logger.error(f"The span can not be negative or 0, got {span}")
            raise ValueError(f"The span can not be negative or 0, got {span}")
        else:
            self.span = span                                    # Section span 
        if angle_units == "deg":
            self.sweep = np.deg2rad(sweep_angle)                # Sweep angle of the section
            self.twist = np.deg2rad(twist_angle)                # Twist angle of the section
            self.dihedral = np.deg2rad(dihedral)                # Dihedral angle of the section
        elif angle_units == "rad":
            self.sweep = sweep_angle
            self.twist = twist_angle
            self.dihedral = dihedral
        else:
            logger.error(f"angle_units got an unexpected argument ({angle_units}). Must be 'deg' or 'rad'")
            raise ValueError(f"angle_units got an unexpected argument ({angle_units}). Must be 'deg' or 'rad'")
        if not 0 <= sweep_loc <= 1:
            logger.error(f"The sweep location should be between 0 and 1 (fraction of the chord)")
            raise ValueError(f"The sweep location should be between 0 and 1 (fraction of the chord)")
        elif not 0 <= twist_loc <= 1:
            logger.error(f"The twist location should be between 0 and 1 (fraction of the chord)")
            raise ValueError(f"The twist location should be between 0 and 1 (fraction of the chord)")
        else:
            self.twistloc = twist_loc
            self.sweeploc = sweep_loc
        self.rootaf = airfoils[0]
        self.tipaf = airfoils[1]

    def apply_twist(self, tip: np.ndarray):
        """
        Function that twists the tip with `self.twist` before interpolation between root and tip
        """



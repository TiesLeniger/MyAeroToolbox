import numpy as np
from ..helpers import rotate_points_about_x_axis, rotate_points_about_y_axis, rotate_points_about_z_axis, rotate_points_about_arbitrary_axis
import logging

logger = logging.getLogger(__name__)
class WingSection:

    def __init__(self, number: int, num_chordwise: int, num_spanwise: int, airfoils: list[object],
                 rootLE: np.ndarray, rootchord: float, taper_ratio: float, span: float, 
                 sweep_angle: float, sweep_loc: float, twist_angle: float, twist_loc: float, 
                 dihedral: float, chordwise_spacing: str = "cosine", angle_units: str = "deg", 
                 parent_section: object = None
                 ):
        self.nc = num_chordwise                             # Number of chordwise panels used to discretise the section
        self.chordspace = chordwise_spacing                 # Chordwise spacing method (constant or cosine)
        self.ns = num_spanwise                              # Number of spanwise panels used to discretise the section
        self.airfoils = airfoils                            # List or tuple of airfoil digits (strings)
        self.rootLE = rootLE                                # [x, y, z] location of the leadig edge in meter
        self.croot = rootchord                              # Root chord in meter
        self.taper = taper_ratio                            # Taper ratio (tip chord / root chord) [-]
        self.ctip = rootchord * taper_ratio                 # Tip chord in meter
        self.span = span                                    # Section span
        if angle_units == "deg":
            self.sweep = np.deg2rad(sweep_angle)            # Sweep angle converted to rad        
            self.twist = np.deg2rad(twist_angle)            # Twist angle converted to rad
            self.dihedral = np.deg2rad(dihedral)            # Dihedral angle converted to rad
        elif angle_units == "rad":
            self.sweep = sweep_angle
            self.twist = twist_angle
            self.dihedral = dihedral                    
        self.sweeploc = sweep_loc                           # Chordwise location at which the sweep angle is specified (0 is LE, 1 is TE)
        self.twistloc = twist_loc                           # Chordwise location at which the twist is applied (0 is LE, 1 is TE)
        self.controlsurfaces = {}                           # Initialise a control surface dict
        if parent_section is not None:                      # Parent section for alignment
            if number == 1:
                logger.warning("Sections with number 1 do not need a parent section")
            else:
                self.parent = parent_section
        elif parent_section is None and number > 1:
            logger.error("Sections with a number higher than 1 must have a parent section for alignment")
            raise ValueError("Sections with a number higher than 1 must have a parent section for alignment")

    def add_controlsurface(self, ID: str, y_bstart: float, y_bstop: float, x_cstart: float, max_up: float, max_down: float):
        pass

    def __align_with_parent(self, num_tol: float = 1e-9):
        if abs(self.parent.ctip - self.croot) > num_tol:
            logger.warning("Parent tip chord is different from root chord. Changing root chord to parent tip chord for alignment")
            self.croot = self.parent.ctip
        if abs(self.parent.twist) > num_tol:
            if not hasattr(self.airfoils[0], 'camberline'):
                self.airfoils[0].calculate_camberline(self.nc, self.chordspace)
            self.airfoils[0].camberline[:, 0] -= self.parent.twist_loc * self.parent.ctip
            self.airfoils[0].camberline = rotate_points_about_y_axis(self.airfoils[0].camberline, self.parent.twist, unit = "rad")
            self.airfoils[0].camberline[:, 0] += self.parent.twist_loc * self.parent.ctip
        if not hasattr(self.parent, 'sectionpoints'):
            self.parent.generate_section_geometry()
        self.rootLE = self.parent.sectionpoints[0, -1, :]
        self.airfoils[0].camberline[0, :] = self.rootLE


    def generate_section_geometry(self):
        pass

                  
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class Airfoil:

    def calculate_camberline(self) -> np.ndarray:
        """
        Calculate the coordinates of the camberline of the airfoil 
        """
        raise NotImplementedError("Subclasses must implement the 'calculate_camberline()' method")
    
    def calculate_airfoil_contour(self) -> np.ndarray:
        """
        Calculate the coordinates of the surface of the airfoil
        """
        raise NotImplementedError("Subclasses must implement the 'calculate_airfoil_contour() method'")

    def plot_camberline(self):
        """
        Plot the camberline
        """
        raise NotImplementedError("Subclasses must impolement the 'plot_camberline()' method")
    
    def plot_airfoil(self):
        """
        Plot the airfoil contour
        """
        raise NotImplementedError("Subclasses must impolement the 'plot_airfoil()' method")
    
class Naca4Digit(Airfoil):

    def __init__(self, NACAdigits: str, chord: float):
        self.m = int(NACAdigits[0])/100                     # Maximum camber in percentage of the chord
        self.p = int(NACAdigits[1])/10                      # Location of maximum camber in tenths of the chord
        self.t = int(NACAdigits[2:])/100                    # Thickness to chord ratio in percentage of the chord
        self.chord = chord                                  # Chord length in meter

    def calculate_camberline(self, numpoints: int, spacing: str = "cosine", opt_return = False) -> np.ndarray:
        """
        Function that calculates 'numpoints' coordinates of the camberline of a NACA 4 digit airfoil
        """
        camberline = np.zeros((numpoints, 3))    

        if spacing == "constant":
            camberline[:, 0] = np.linspace(0.0, 1.0, num = numpoints)
        elif spacing == "cosine":
            psi = np.linspace(0.0, np.pi, num = numpoints)
            camberline[:, 0] = 1/2*(1-np.cos(psi))
        else:
            logger.error(f"Specified spacing method, {spacing}, is not implemented. Choose 'constant' or 'cosine'")
            raise ValueError(f"Specified spacing method, {spacing}, is not implemented. Choose 'constant' or 'cosine'")
        
        if abs(self.p) > 1e-9:
            before_max_camber = camberline[:, 0] < self.p
            after_max_camber = ~before_max_camber
            camberline[before_max_camber, 2] = (self.m/(self.p**2))*(2*self.p*camberline[before_max_camber, 0] - camberline[before_max_camber, 0]**2)
            camberline[after_max_camber, 2] = (self.m/(1-self.p**2))*(1 - 2*self.p + 2*self.p*camberline[after_max_camber, 0] - camberline[after_max_camber, 0]**2)
        
        camberline *= self.chord
        self.camberline = camberline

        if opt_return:
            return camberline
        
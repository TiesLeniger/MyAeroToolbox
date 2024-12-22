import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

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
            before_max_camber = camberline[:, 0] <= self.p
            after_max_camber = ~before_max_camber
            camberline[before_max_camber, 2] = (self.m/self.p**2)*(2*self.p*camberline[before_max_camber, 0] - camberline[before_max_camber, 0]**2)
            camberline[after_max_camber, 2] = (self.m/(1-self.p)**2)*(1 - 2*self.p + 2*self.p*camberline[after_max_camber, 0] - camberline[after_max_camber, 0]**2)
        
        camberline *= self.chord
        self.camberline = camberline

        if opt_return:
            return camberline

    def calculate_airfoil_contour(self, numpoints: int, spacing: str = "cosine", close_TE: bool = False, opt_return: bool = False) -> np.ndarray:
        """
        """
        airfoil_contour = np.zeros((numpoints, 3))
        if spacing == "constant":
            top = np.linspace(0.0, 1.0, num = np.ceil(numpoints/2))
            bot = np.linspace(1.0, 0.0, num = np.floor(numpoints/2))
            airfoil_contour[:, 0] = np.concatenate((top, bot[1:]))
            LEindex = np.where(airfoil_contour[:, 0] == 0.0)[0][0]
        elif spacing == "cosine":
            psi = np.linspace(0.0, 2*np.pi, num = numpoints)
            airfoil_contour[:, 0] = 1/2*(1+np.cos(psi))
            LEindex = np.argmin(airfoil_contour[:, 0])
        else:  
            logger.error(f"Specified spacing method, {spacing}, is not implemented. Choose 'constant' or 'cosine'")
            raise ValueError(f"Specified spacing method, {spacing}, is not implemented. Choose 'constant' or 'cosine'")
        
        if abs(self.p) > 1e-9:
            before_max_camber = airfoil_contour[:, 0] <= self.p
            after_max_camber = ~before_max_camber
            airfoil_contour[before_max_camber, 2] = (self.m/self.p**2)*(2*self.p*airfoil_contour[before_max_camber, 0] - airfoil_contour[before_max_camber, 0]**2)
            airfoil_contour[after_max_camber, 2] = (self.m/(1-self.p)**2)*(1 - 2*self.p + 2*self.p*airfoil_contour[after_max_camber, 0] - airfoil_contour[after_max_camber, 0]**2)
        
        # Gradient of the camberline
        gradient = np.zeros((numpoints))
        gradient[before_max_camber] = (2*self.m/self.p**2)*(self.p - airfoil_contour[before_max_camber, 0])
        gradient[after_max_camber] = (2*self.m/(1-self.p)**2)*(self.p - airfoil_contour[after_max_camber, 0])

        # Polynomial coefficients for thickness distribution
        a = [0.2969, -0.1260, -0.3516,  0.2843, -0.1015]
        if close_TE:
            a[-1] = -0.1036
        thickness_distribution = (self.t/0.2)*(a[0]*np.sqrt(airfoil_contour[:, 0]) + a[1]*airfoil_contour[:, 0] + a[2]*airfoil_contour[:, 0]**2 + a[3]*airfoil_contour[:, 0]**3 + a[4]*airfoil_contour[:, 0]**4)

        # Angle of the camberline
        theta = np.arctan(gradient)

        # Calculate the airfoil contour
        airfoil_contour[:LEindex, 0] = airfoil_contour[:LEindex, 0] - thickness_distribution[:LEindex]*np.sin(theta[:LEindex])
        airfoil_contour[:LEindex, 2] = airfoil_contour[:LEindex, 2] + thickness_distribution[:LEindex]*np.cos(theta[:LEindex])
        airfoil_contour[LEindex:, 0] = airfoil_contour[LEindex:, 0] + thickness_distribution[LEindex:]*np.sin(theta[LEindex:])
        airfoil_contour[LEindex:, 2] = airfoil_contour[LEindex:, 2] - thickness_distribution[LEindex:]*np.cos(theta[LEindex:])

        # Scale the airfoil contour with the chord
        airfoil_contour *= self.chord

        # Save the airfoil contour
        self.airfoil_contour = airfoil_contour

        # Return the airfoil contour if specified
        if opt_return:
            return airfoil_contour

    def plot_camberline(self):
        """
        Plot
        """
        if not hasattr(self, 'camberline'):
            logger.error("Camberline not calculated. Run 'calculate_camberline()' first")
            raise AttributeError("Camberline not calculated. Run 'calculate_camberline()' first")
        plt.figure()
        plt.plot(self.camberline[:, 0], self.camberline[:, 2], color = "blue")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        name = str(int(self.m*100)) + str(int(self.p*10)) + str(int(self.t*100))
        plt.title(f"Camberline - NACA {name}")
        plt.grid()
        plt.axis("equal")
        path_to_output = Path.cwd() / "output" / "plots" / f"camberline-NACA-{name}.png"
        plt.savefig(path_to_output)
        plt.close()

    def plot_airfoil(self): 

        if not hasattr(self, 'airfoil_contour'):
            logger.error("Airfoil contour not calculated. Run 'calculate_airfoil_contour()' first")
            raise AttributeError("Airfoil contour not calculated. Run 'calculate_airfoil_contour()' first")
        plt.figure()
        plt.plot(self.airfoil_contour[:, 0], self.airfoil_contour[:, 2], color = "blue")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        name = str(int(self.m*100)) + str(int(self.p*10)) + str(int(self.t*100))
        plt.title(f"Airfoil contour - NACA {name}")
        plt.grid()
        plt.axis("equal")
        path_to_output = Path.cwd() / "output" / "plots" / f"airfoil-NACA-{name}.png"
        plt.savefig(path_to_output)
        plt.close()


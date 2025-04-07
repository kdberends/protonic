import numpy as np 
from collections import namedtuple
import constants as const

Material = namedtuple('Material', ['Z', 'A', 'I', 'rho'])
Compound = namedtuple('Compound', ['elements', 'weights', 'rho'])
elements = {
    "H": Material(Z=1, A=1, I=19.2e-6, rho = 8.37480E-05), 
    "HE": Material(Z=2, A=4, I=41.8e-6, rho=1.66322E-04),
    "Fe": Material(Z=26, A = 55.845, I = 286e-6, rho=7.874),
    "O": Material(Z=8, A=15.999, I=95e-6, rho= 1.33151E-03),
    "air": Material(Z=7, A=28, I=85e-6, rho=1.225E-03), # move to compounds https://physics.nist.gov/cgi-bin/Star/compos.pl?refer=ap&matno=104
}


compounds = {
    "H2O": Compound(elements=[elements.get("H"), elements.get("O")], weights=[0.111894, 0.888106], rho=1.000)
    }


Projectile = namedtuple('Projectile', ['E0',            # rest mass in MeV
                                       'm0',            # rest mass in kg
                                       'z'              # particle charge in e
                                       ])

projectiles = {
    "proton": Projectile(E0=938.27208943, m0=1.67262e-27, z=1),
    "alpha": Projectile(E0=3727.379, m0=6.644656e-27, z=2),
}



def PSTAR(material="H"):
    """
    Read PSTAR data for a given material.
    The data is stored in a file named "pstar{material}.dat" in the "../data/" directory.
    The file format is as follows:
    - The first 8 lines are header information.
    - The data starts from the 9th line onwards.
    - The data is space-separated.
    - The first column is the energy in MeV.
    - The second column is the stopping power in MeV cm^2/g.
    - The third column is the range in g/cm^2.
    """
    with open(f"../data/pstar{material}.dat", "r") as f:
        lines = [line.strip() for line in f]
        data = np.loadtxt(lines, delimiter=' ', skiprows=8)
    return data


def bethebloch(Ek, material:Material|Compound, projectile:Projectile) -> float:
    """
    Calculate the Bethe-Bloch formula for a given energy, material, and projectile.
    Ek: Kinetic energy in MeV
    material: Material object or list of Material objects
    projectile: Projectile object
    """

    beta = lambda Ek:  (1-(Ek/projectile.E0 + 1)**-2)**0.5
    K = 4*np.pi*const.Na*const.re**2*const.mec2  # MeV cm2 / g
    logterm = lambda material: np.log(2*const.mec2*beta(Ek)**2/(material.I*(1-beta(Ek)**2)))
    bethe = lambda Ek, material, projectile: K * projectile.z**2 * material.Z / material.A * 1 / beta(Ek)**2 * (logterm(material) - beta(Ek)**2)
    bethe_compound = lambda x, compound, projectile: np.sum([weight*bethe(x, element, projectile) for element, weight in zip(compound.elements, compound.weights)], axis=0) 
    
    if isinstance(material, Material):
        return bethe(Ek, material, projectile)
    elif isinstance(material, Compound):
        return bethe_compound(Ek, material, projectile)
    

if __name__ == "__main__":
    # Example usage
    material = elements.get("H")
    projectile = projectiles.get("proton")
    Ek = 1.0  # Kinetic energy in MeV
    print(bethebloch(Ek, material, projectile))
    
    # Example usage for compound
    compound = compounds.get("H2O")

    print(bethebloch(Ek, compound, projectile))
    # Example usage for PSTAR
    # material = "H"
    # data = PSTAR(material)
    # print(data)

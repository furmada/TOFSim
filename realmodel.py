import numpy as np
from libem import ChargedParticle3D

class Conversion(object):
    def __init__(self, **constants):
        """
        Define a simulation to real unit conversion using the scheme
        <SIM> * [CONST] = <REAL>
        """
        self.constants = {
            "mass": 1.0,
            "length": 1.0,
            "time": 1.0,
            "charge": 1.0,
            "voltage": 1.0
        }
        self.constants.update(**constants)

    def sim_to_real(self, value, utype="length"):
        return value * eval(utype, self.constants)

    def real_to_sim(self, value, utype="length"):
        return value / eval(utype, self.constants)
    
    def superimpose(self, conv):
        joined = {}
        for factor in set(self.constants.keys()).union(set(conv.constants.keys())):
            if factor == "__builtins__": continue
            joined[factor] = self.constants.get(factor, 1.0) * conv.constants.get(factor, 1.0)
        return Conversion(**joined)

class SimIonBeam(object):
    def __init__(self, path, conversions=None, skip_rows=10, axis_map=("Z", "X", "Y")):
        self.conversion = Conversion() if conversions is None else conversions
        self.data = np.genfromtxt(path, delimiter=",", skip_header=skip_rows, names=True)
        self.axis_map = axis_map

    def sample(self, num_particles=1, assign_sim=None, real_charge=1, radius=0, gravity=-1, bounce=None, track_force=False):
        particles = []
        radius = self.conversion.real_to_sim(radius, "length")
        for _ in range(num_particles):
            s = self.data[np.random.randint(0, self.data.shape[0])]
            mass = self.conversion.real_to_sim(s["Mass"], "mass")
            charge = self.conversion.real_to_sim(real_charge, "charge")
            position = self.conversion.real_to_sim(np.array([s[self.axis_map[i]] for i in range(3)]), "length")
            velocity = self.conversion.real_to_sim(np.array([s["V" + self.axis_map[i].lower()] for i in range(3)]), "length / time") 
            particles.append(ChargedParticle3D(assign_sim, mass, charge, position, velocity, radius, gravity, bounce, track_force))
        return particles
            

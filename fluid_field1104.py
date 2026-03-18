# fluid_field.py
import numpy as np

class FluidField:
    """
    Single fluid body: Directly returns constant (rho, Cf, CD, CA, CM)
    Two-phase flow: Defines φ(x) and smoothing σ(φ)=0.5(1+tanh(φ/eps)), blends parameters from both sides
    """
    def __init__(self, single_fluid=True, params1=None, params2=None, eps=0.02):
        self.single = single_fluid
        self.eps = float(eps)

        # Default fluid parameters
        def _pack(p):  # rho, Cf, CD, CA, CM
            return dict(rho=p.get("rho", 1000.0),
                        Cf=p.get("Cf", 0.03),
                        CD=p.get("CD", 2.0),
                        CA=p.get("CA", 1.0),
                        CM=p.get("CM", 1.0))
        # p1=water
        self.p1 = _pack(params1 or {})
        
        # p2=air
        self.p2 = _pack(params2 or {
            "rho": 1.225,    # Air density (kg/m³)
            "Cf": 0.005,     # Air friction coefficient (much smaller than water)
            "CD": 1.0,       # Air drag coefficient
            "CA": 0.5,       # Air added mass coefficient
            "CM": 0.5        # Air rotational added mass coefficient
        })

    def phi(self, x):
        """
        Signed distance function: >0 is fluid 1, <0 is fluid 2.
        Placeholder: default single fluid -> always +1; you can replace this with the actual interface expression later.
        """
        return -x[2]

    def blend(self, s, a, b):
        return s * a + (1.0 - s) * b

    def at(self, x_world3):
        """
        Returns the fluid parameters at this spatial point
        """
        if self.single:
            return self.p1
        phi = self.phi(np.asarray(x_world3))
        s = 0.5 * (1.0 + np.tanh(phi / self.eps))  # Smooth blending
        out = {}
        for k in self.p1.keys():
            out[k] = self.blend(s, self.p1[k], self.p2[k])
        return out

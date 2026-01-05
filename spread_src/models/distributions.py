"""
Custom NGBoost distributions for training and inference.

SafeT: Student-T distribution with clipped scale/df to prevent numerical overflow.
"""

import numpy as np
from ngboost.distns import T
from scipy.stats import t as t_dist


class SafeT(T):
    """
    Student-T distribution with safe parameter bounds to prevent numerical overflow.
    
    Clips params BEFORE exp() to prevent overflow:
    - log-scale clipped to [-7, 10] → scale in [0.001, 22026]
    - log-df clipped to [log(2), log(30)] → df in [2, 30]
    
    This class must be importable from the same location when loading pickled models.
    """
    def __init__(self, params):
        # Clip log-scale (params[1]) to a safe range BEFORE exp()
        # Lower: np.log(1e-3) ~ -6.9, Upper: 10.0 → exp(10) ~ 22,000
        params[1] = np.clip(params[1], np.log(1e-3), 10.0)
        
        # Clip log-df (params[2])
        # df between 2.0 (fat tails) and 30.0 (near normal)
        params[2] = np.clip(params[2], np.log(2.0), np.log(30.0))
        
        # Store clipped params
        self._params = params
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.var = self.scale ** 2
        self.df = np.exp(params[2])
        
        # Initialize the underlying scipy distribution
        self.dist = t_dist(df=self.df, loc=self.loc, scale=self.scale)

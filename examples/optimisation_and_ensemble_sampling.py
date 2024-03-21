import numpy as np 
import os

from operator import itemgetter

import matplotlib.pyplot as plt 
import corner 

import PySO

import time


class Gaussian2D(PySO.Model):
    """
    A simple 2D Gaussian distribution
    """    
    dim = 2
    names    = ['x0', 'x1'] 
    bounds   = [[-4.,4.], [-4.,4.]]
    periodic = [0, 0]

    def __init__(self, mean=None, cov=None):
        """
        INPUTS
        ------
   t_start = time.time()    mean: ndarray, shape=(2,)
            the mean
        cov: ndarray, shape=(2,2)
            the covariance matrix
        """
        if mean is None:
            self.mean = np.zeros(self.dim)
        else:
            assert mean.shape==(self.dim,)
            self.mean = np.array(mean)

        if cov is None:
            self.cov = np.eye(self.dim)
        else:
            assert cov.shape==(self.dim, self.dim)
            self.cov = cov

        self.invcov = np.linalg.inv(self.cov)
    
    def log_likelihood(self, params):
        """
        The unnormalised log PDF of the dist

        INPUTS
        ------
        params: dict
            the parameters of the distribution
            keys must include the entries in list self.names
        
        RETURNS
        -------
        ans: float
            the log PDF
        """
        x = np.array(itemgetter(*self.names)(params))
        ans = -0.5*np.einsum('i,ij,j->', x, self.invcov, x)
        return ans

def main():

    dist = Gaussian2D()

    NumParticles = 2000

    outdir = 'ensemble_demo'

    myswarm = PySO.Swarm(dist,
                        NumParticles,
                        Output = outdir,
                        Verbose = True,
                        Nperiodiccheckpoint = 1, # Final two args mean evolution is saved at every iteration. Only necessary if running myswarm.Plot()
                        Saveevolution = True,    ############
                        Nthreads=8,
                        Tol = 1.0e-2,
                        #Omega = 0.5, Phip = 0.6, Phig = 0.6, 
                        Omega = 0., Phip = 0., Phig = 0., Mh_fraction=1.,
                        Maxiter=20,)

    # Clear any existing history file
    history_file = os.path.join(outdir, "SwarmEvolutionHistory.dat")
    if os.path.isfile(history_file): os.system('rm {}'.format(history_file))

    t_start = time.time()
    myswarm.Run()
    t_end = time.time()
    
    print("evolution took {} seconds".format(t_end-t_start))

    myswarm.PlotSwarmEvolution()

    return myswarm



if __name__=='__main__':

    myswarm = main()

    print(np.mean(myswarm.Points, axis=0))
    print(np.cov(myswarm.Points.T))

    corner.corner(myswarm.Points, labels=myswarm.Model.names)
    plt.show()
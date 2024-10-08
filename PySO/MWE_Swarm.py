import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from PySO.affine_invariant_utils import sample_g, ParallelStretchMove_InternalFunction
import scipy 

try:
    print('Defaulting to torch.multiprocessing')
    from torch.multiprocessing import Pool, set_start_method
    set_start_method('spawn',force=True)
except: 
    print('PyTorch not installed, not using torch.multiprocessing, using pathos.multiprocessing instead')
    from pathos.multiprocessing import ProcessingPool as Pool

class Swarm(object):

    def __init__(self,
                 Model,
                 NumParticles,
                 Omega = 0.6,        # PSO rule parameter
                 Phip = 0.2,         # PSO rule parameter
                 Phig = 0.2,        # PSO rule parameter
                 Mh_fraction= 0.0,
                 Jitter_weight = 0.0,
                 Tol = 1.0e-3,
                 Automatic_convergence_testing = False,
                 Convergence_testing_num_iterations = 50,
                 Maxiter = 1.0e6,
                 Periodic = None,
                 Initialguess = None,
                 Initial_placement = 'LHS',
                 Nthreads = None,
                 Provided_pool = None, # Pool object for multiprocessing
                 Seed = None,
                 Nperiodiccheckpoint = 10,
                 Output = './',
                 Resume = False,
                 Verbose = False,
                 Saveevolution = False,
                 Plotevolution = False,
                 Velocity_min = None,
                 Velocity_minimum_factor = 100,
                 Proposalcov = None,
                 Initial_guess_v_factor = 3,
                 Clip_lower_velocity = True,
                 Clip_upper_velocity = True,
                 Delta_max = 0.3,
                 Delta_min = 0.0001,
                 Velocity_clipping_or_rescale= 'Clip',
                 Reinitialise_velocities_from_initial_guess=True,
                 Affine_invariant_a = 2.0,
                 Constriction = False,
                 Constriction_kappa = 1.0):
        """

        Minimum working example of Particle swarm optimization class.


        REQUIRED INPUTS
        ------

        Model: object inheriting PySO.Model.Model
            this defines the problem
        NumParticles: int
            number of particles in the swarm


        OPTIONAL INPUTS
        ---------------

        (Algorithm parameters)
        Omega: float
            the omega parameter, inertial coefficient for velocity updating [defaults to .6]
        Phip: float
            the phi_p parameter, cognitive coefficient for velocity updating [defaults to .2]
        Phig: float
            the phi_g parameter, social coefficient for velocity updating [defaults to .2]
        Mh_fraction: float
            parameter controlling proportion of velocity rule dictated by MCMC [defaults to 0.]
        Jitter_weight: float
            parameter that controls the movement of velocities to point towards a random particle in the swarm [defaults to 0.]
             (used to get particle out of local maxima they are stuck in)
        Tol: float
            the minimum improvement on functionvalue that we class as "still converging"
        Automatic_convergence_testing: boolean
            Flag toggling convergence testing to terminate swarm evolution [defaults to False]
        Convergence_testing_num_iterations: int
            If best swarm point has not improved over this many last iterations (improved past Tol) [defaults to 50]
        Maxiter: int
            the maximum number of evolution iterations [defaults to 1.0e6]
        Periodic: list [defaults to None]
            In which dimensions to use periodic boundary conditions. E.g. [0,1,0,...,0] means periodic in dimension 1 only.
            If None, then use all zeros.
        Initialguess: list of arrays or dicts
            optionally provide initial guess(es) [defaults to None]
            list structure [p1, p2, ..., pm]
            the first m swarm points will start at these locations
            p1,p2... must be arrays of length Ndim whose entries correspond to Model.names
            or p1,p2... must be dicts with keys Model.names
        Initial_placement: str (Either 'LHS' or 'Random')
            Method to place initial guesses in the swarm, choices are 'Random' for random sampling and 'LHS' for Latin Hypercube Sampling [defaults to 'LHS']


        (Other parameters)
        Nthreads: int [defaults to None]
            Number of multiprocessing threads to use. If None, use a serial version.
        Provided_pool: Pool object [Defaults to None]
            Pool object for multiprocessing. If pool not provided, creates a new pool. If pool provided, uses that pool.
        Seed: int
            random Seed [defaults to None]
        Nperiodiccheckpoint: int
            number of iterations between checkpoints [defaults to 10]
        Output: str
            folder in which to save output [defaults to './']
        Resume: bool
            look for resume pkl file on startup and checkpoint during run [defaults to False]
        Verbose: bool
            Verbosity [defaults to False]
        Saveevolution: bool
            save the entire evolution of the swarm [defaults to False]
        Plotevolution: bool
            Plot the function values as a function of iteration and the pairplot trajectories for particles [defaults to False]
        Velocity_min: None or numpy array
            (absolute) minimum velocity in every dimension, defaults to 1/100th of each dimension
        Velocity_minimum_factor: int or float
            Set absolute minimum speed to 1/velocity_minimum_factor of the prior range specified in each parameter,
             Only relevant of velocity min array not provided by user [defaults to 100]
        Proposalcov: numpy array
            Covariance matrix for gaussian proposal distribution for MH part of velocity rule [defaults to identity]
        Initial_guess_v_factor: float/int
            Fiddle factor used to multiply initial guess spreads to initialise velocities [defaults to 3]
        Clip_lower_velocity: boolean
            Flag for if velocity should be clipped (rescaled) on the lower end preserving direction of travel [defaults to True]
        Clip_upper_velocity: boolean
            Flag for if velocity should be clipped (rescaled) on the upper end preserving direction of travel [defaults to True]
        Delta_max: float
            fiddle parameter used to control maximum magnitude of velocity vector [defaults to 0.3] [Only used if velocity rescaling enabled]
        Delta_min: float
            fiddle parameter used to control minimum mangitude of velocity vector [defaults to 0.0001] [Only used if velocity rescaling enabled]
        Velocity_clipping_or_rescale: str, either 'Clip' or 'Rescale'
            flag used to choose prescription for clamping velocities, either clip velocities to some minimum velocity in each dimension, or
             rescale to some minimum and or maximum velocity magnitudes.
        Reinitialise_velocities_from_initial_guess: boolean
            Wether to reinitialise velocities from initial guess (or if false, initialise from v_min array provided by user) [defaults to True]
        affine_invariant_a: float
            Parameter for the affine invariant MCMC velocity rule. Controls how large of a step the sampler takes, 
                Acceptance rate goes down if a goes up [defaults to 2.0]
        Constriction: Boolean
            Flag to use constriction factor in PSO velocity rule [defaults to True]
                Constriction factor derived in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=985692.
        Constriction_kappa: float
            Constriction factor kappa [defaults to 1.0], Only used when Constriction is True, derived in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=985692.
                Controls degree of convergence of the swarm. 
                Note PhiP=PhiG=2, Kappa=1 gives the same velocity rule as the standard PSO rule.
        """
        
        self.Ndim = len(Model.names)

        self.Model = Model

        self.NumParticles = NumParticles

        self.Points = np.zeros( (NumParticles, self.Ndim) )
        self.Values = np.zeros(NumParticles)

        self.BestKnownPoints = np.zeros( (NumParticles, self.Ndim) )
        self.BestKnownValues = np.zeros(NumParticles)

        self.BestKnownSwarmPoint = np.zeros(self.Ndim)
        self.BestKnownSwarmValue = 0.0

        self.Velocities = np.zeros( (NumParticles, self.Ndim) )

        self.Tol = Tol
        self.Convergence_testing_num_iterations = Convergence_testing_num_iterations

        self.initial_guess_v_factor = Initial_guess_v_factor

        self.Omega = Omega
        self.PhiP = Phip
        self.PhiG = Phig
        self.MH_fraction = Mh_fraction
        self.Jitter_weight  = Jitter_weight


        if Proposalcov is None:
            self.ProposalCov = np.identity(self.Ndim)
        else:
            self.ProposalCov = Proposalcov

        self.EvolutionCounter = 0
        self.MaxIter = Maxiter

        self.InitialGuess = Initialguess
        
        # Initial placement method
        self.Initial_placement = Initial_placement

        self.Seed = Seed

        self.nPeriodicCheckpoint = Nperiodiccheckpoint

        self.Output = Output


        self.Resume = Resume

        self.Verbose = Verbose

        self.SaveEvolution = Saveevolution
        self.Plotevolution = Plotevolution


        self.Nthreads = Nthreads

        # If Pool is not provided, create a new Pool
        if Provided_pool is None and self.Nthreads is not None:
            self.parallel = True
            self.Pool = Pool(self.Nthreads)
        if Provided_pool is not None:
            self.parallel = True
            self.Pool = Provided_pool
        if self.Nthreads is None and Provided_pool is None:
            self.parallel = False

        if Periodic is None:
            self.Periodic = np.array([ 0 for i in range(self.Ndim)])
        else:
            assert (  len(Periodic)==self.Ndim  and  all(np.isin(Periodic, [0,1]))  )
            self.Periodic = np.array(list(Periodic))

        self.PeriodicParamRanges = np.array([
                                            np.inf if self.Periodic[i]==0 else np.ptp(b)
                                        for i, b in enumerate(self.Model.bounds)])

    # Param indexes where periodicity happens
        self.Periodic_params = np.where(self.Periodic == 1)[0]

        self.BoundsArray = np.array(self.Model.bounds)

        #     NOTE: ptp by default positive


        if Velocity_clipping_or_rescale == 'Clip':
            self.velocity_clipping_function = self.velocity_clipping
            self.velocity_minimum_factor = Velocity_minimum_factor
            self.velocity_min = Velocity_min
            if np.all(self.velocity_min == None):
                self.velocity_min = np.ptp(self.BoundsArray, axis=1) / self.velocity_minimum_factor
    
        elif Velocity_clipping_or_rescale == 'Rescale':

            self.velocity_clipping_function = self.rescale_velocities
            self.delta_max = Delta_max
            self.delta_min = Delta_min
            self.velocity_max = self.delta_max*np.linalg.norm(np.ptp(self.BoundsArray,axis=1))
            self.velocity_min = self.delta_min*np.linalg.norm(np.ptp(self.BoundsArray,axis=1))
            self.clip_lower_velocity = Clip_lower_velocity
            self.clip_upper_velocity = Clip_upper_velocity

        # Whether to reinitialise velocities from initial guess or from v_min array provided by user
        self.reinitialise_velocities_from_initial_guess = Reinitialise_velocities_from_initial_guess


        if self.MH_fraction == 0:
            # The velocity rule is the PSO standard rule when no MH
            self.VelocityRule = self.PSO_VelocityRule
        elif self.MH_fraction == 1: 
            # The velocity rule is the Affine Invariant MC rule
            self.VelocityRule = self.ParallelAffineInvariantMC_VelocityRule_
        else:
            # Some combinaton of the 2
            self.VelocityRule = self.Hybrid_VelocityRule


        # Only used for hierarchical PSO searches
        self.Hierarchical_step = 0
        self.Spreads = []
        self.FuncHistory = []

        if Automatic_convergence_testing == True:
            self.ContinueCondition = self.ContinueCondition_Hybrid
        else:
            self.ContinueCondition = self.ContinueCondition_Vanilla
        self.log_posterior = None        

        # Only used for the affine invariant MCMC velocity rule
        self.affine_invariant_a = Affine_invariant_a    

        if Constriction == True:
            Phi = self.PhiP + self.PhiG
            if Phi < 4: 
                assert False, "PhiP + PhiG must be greater than 4 for the constriction factor to work"
            self.Constriction_factor = (2*Constriction_kappa)/np.abs(2-(Phi)-np.sqrt(Phi**2 -4*Phi))
        else: 
            self.Constriction_factor = 1
    def MyFunc(self,p):
        """
        The function to be maximised.
        Converts array into dictionary and calls self.Model.log_posterior

        INPUTS
        ------
        p: array
            Array of parameters. Order matches that in self.Model.names

        RETURNS
        -------
        log_posterior: float
        """
        par_dict = dict(zip(self.Model.names, p))
        self.log_posterior = self.Model.log_likelihood(par_dict)
        return self.log_posterior


    def InitialiseSwarm(self):
        """
        Initialise the swarm points, values and velocities
        """
        resume_file = os.path.join(self.Output, "PySO_resume.pkl")

        if self.Resume and os.path.exists(resume_file):

            print('Resuming from file {}'.format(resume_file))
            self.ResumeFromCheckpoint()

        else:

            # Initialise counter and random seed
            self.EvolutionCounter = 0
            if self.Seed is not None:
                np.random.seed(seed=self.Seed)

            # Initialise the particle positions/velocities 

            if self.Initial_placement == 'Random':
                for i in range(self.NumParticles):
                    for j in range(self.Ndim):

                        Low, High = self.Model.bounds[j]
                        self.Points[i][j] = np.random.uniform(Low, High)

                        Range = abs(High-Low)
                        vLow, vHigh = -Range/5., Range/5. #FIDDLE PARAMETER HERE
                        self.Velocities[i][j] = np.random.uniform(vLow, vHigh)

            elif self.Initial_placement == 'LHS':
                initial_sampler = scipy.stats.qmc.LatinHypercube(self.Ndim,strength=1)
                self.Points = initial_sampler.random(n=self.NumParticles)*(np.ptp(np.array(self.Model.bounds),axis=1)) + np.array(self.Model.bounds)[:,0]

                # Still initialise velocities randomly with a fiddle factor. 
                for i in range(self.NumParticles):
                    for j in range(self.Ndim):

                        Low, High = self.Model.bounds[j]

                        Range = abs(High-Low)
                        vLow, vHigh = -Range/5., Range/5. #FIDDLE PARAMETER HERE
                        self.Velocities[i][j] = np.random.uniform(vLow, vHigh)


            # If provided, overwrite first M positions with initial guesses
            if self.InitialGuess is not None:
                M = len(self.InitialGuess)
                assert M==self.NumParticles
                for i in range(M):

                    if isinstance(self.InitialGuess[i], (np.ndarray)):
                        self.Points[i] = self.InitialGuess[i]
                    elif isinstance(self.InitialGuess[i], (dict)):
                        self.Points[i] = np.array([ self.InitialGuess[i][name] for name in self.Model.names ])
                # If initial guess provided, velocities drawn from spread of initial distribution
                # Spread in velocities according to the normal distribution specified by the covariance of initial positions
                cov = np.cov(self.Points.T)
                self.Velocities = np.random.multivariate_normal(np.zeros(self.Ndim), cov, size=self.NumParticles)*self.initial_guess_v_factor/np.sqrt(self.Ndim)

                if self.reinitialise_velocities_from_initial_guess == True:
                    if self.velocity_clipping_function == self.rescale_velocities:
                        # set up velocity rescaling bounds from initial distribution if using the rescaling
                        self.velocity_max = self.delta_max * np.linalg.norm(np.ptp(self.Points, axis=0))
                        self.velocity_min = self.delta_min * np.linalg.norm(np.ptp(self.Points, axis=0))
                    elif self.velocity_clipping_function == self.velocity_clipping:
                        # set up minimum velocity bounds using initial distribution if using the rescaling
                        self.velocity_min = np.ptp(self.Points, axis=0) / self.velocity_minimum_factor
                else:
                    # v_min already set in init method, above lines overwrite if set to initialise velocities from initial guess.
                    pass

            # Initialise the particle function values
            if self.parallel:
                self.Values = np.array( self.Pool.map(self.MyFunc, self.Points) )
            else: 
                self.Values = np.array( list(map(self.MyFunc, self.Points)) )

            # Initialise each particle's best known position to initial position
            self.BestKnownPoints = self.Points.copy()
            self.BestKnownValues = self.Values.copy()

            # Update the swarm's best known position
            self.BestKnownSwarmPoint = self.BestKnownPoints[np.argmax(self.BestKnownValues)]

            self.BestKnownSwarmValue = np.max(self.BestKnownValues)

            # Calculate initial function value spread for switch condition
            self.Spreads.append(np.ptp(self.Values))

    def EnforceBoundaries(self):
        """
        Boundary conditions on the edge of the search region
        """
        # Periodic boundary conditions
        self.Points[:,self.Periodic_params] = self.BoundsArray[self.Periodic_params,0] + np.mod(self.Points[:,self.Periodic_params]-self.BoundsArray[self.Periodic_params,0],self.PeriodicParamRanges[self.Periodic_params])

        # Mask for points that are out of range
        upper_mask_indices = np.where((self.Points - self.BoundsArray[:,1])>0)
        lower_mask_indices = np.where((self.Points - self.BoundsArray[:,0])<0)

        # Reflective boundary conditions (positions)
        self.Points[upper_mask_indices] = self.BoundsArray[upper_mask_indices[1],1] - np.abs(self.Points[upper_mask_indices] - self.BoundsArray[upper_mask_indices[1],1])
        self.Points[lower_mask_indices] = self.BoundsArray[lower_mask_indices[1],0] + np.abs(self.Points[lower_mask_indices] - self.BoundsArray[lower_mask_indices[1],0])

        # Clipping positions (to make extra sure that all params are in range)
        self.Points = np.clip(self.Points, a_min=self.BoundsArray[:,0], a_max=self.BoundsArray[:,1])

        # Reflective boundary conditions (velocities)
        self.Velocities[upper_mask_indices] *= -1
        self.Velocities[lower_mask_indices] *= -1



    def PSO_VelocityRule(self):
        """
        The standard PSO rule for updating the velocities with a jitter component added to reduce the chance of a particle
        stuck in a local maxima

        RETURN:
        ------
        clipped_velocities: numpy array
            velocities clipped to the minimum velocity for each dimension

        """
        best_known_swarm_point = np.tile(
                              self.BestKnownSwarmPoint, self.NumParticles
                                  ).reshape((self.NumParticles, self.Ndim))
        unclipped_velocities = self.Constriction_factor*(self.Omega * self.Velocities
               + self.PhiP * np.random.uniform(size=(self.NumParticles,self.Ndim)) * ( self.BestKnownPoints - self.Points )
               + self.PhiG * np.random.uniform(size=(self.NumParticles,self.Ndim)) * ( best_known_swarm_point - self.Points)
               # line "jitters" each particle towards another random particle in the swarm to prevent being stuck in local maxima
               + self.Jitter_weight * np.random.uniform(size=(self.NumParticles,self.Ndim)) * (self.Points[np.random.randint(self.NumParticles,size=self.NumParticles)]-
                                                                                               self.Points))
        clipped_velocities = self.velocity_clipping_function(unclipped_velocities)
        return (clipped_velocities)

    def velocity_clipping(self,unclipped_velocities):
        """
        Clip the velocities only at the lower end, an alternative to velocity rescaling method

        Use this method in problems where you REALLY dont want the velocity for 1 or more parameters to be too small as this
            is possible with the resclaing method
        INPUT:
        ----------
        unclipped_velocities: ndarray (#Num_particles,#Ndim)
            PSO velocities unclipped

        RETURN:
        -------
        clipped_velocities: ndarray (#Num_particles,#Ndim)
            PSO velocities clipped
        """
        clipped_velocities = np.sign(unclipped_velocities)*np.clip(np.abs(unclipped_velocities), a_min = self.velocity_min, a_max= None)
        return(clipped_velocities)

    def rescale_velocities(self,unclipped_velocities):
        """
        Rescale velocities (on the upper  and lower end) by magnitude preserving particle directions.
        (See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8280887)

        INPUT:
        ----------
        unclipped_velocities: ndarray (#Num_particles,#Ndim)
            PSO velocities unclipped

        RETURN:
        -------
        clipped_velocities: ndarray (#Num_particles,#Ndim)
            PSO velocities rescaled
        """
        # Calculate velocity magnitudes for each particle
        velocity_magnitudes_uncliped = np.linalg.norm(np.abs(unclipped_velocities),axis=1)

        # Clip manitudes by maximum mangitudes and minimum and detect particles where cliping was applied
        # Seperate the lower and upper edge clipping
        #    //TODO: probably can clean this up into one expression but not urgent
        velocity_magnitudes_clipped_upper= np.clip(velocity_magnitudes_uncliped, a_min = None, a_max=self.velocity_max)
        velocity_clipping_indices_upper = np.where(velocity_magnitudes_uncliped != velocity_magnitudes_clipped_upper)[0]

        velocity_magnitudes_clipped_lower= np.clip(velocity_magnitudes_uncliped, a_min = self.velocity_min, a_max= None)
        velocity_clipping_indices_lower = np.where(velocity_magnitudes_uncliped != velocity_magnitudes_clipped_lower)[0]

        print('Number of velocity magnitudes rescaled (lower,upper): ',(velocity_clipping_indices_lower.size,velocity_clipping_indices_upper.size))
        # Clipped velocities will be of the same shape as unclipped velocities
        clipped_velocities = unclipped_velocities.copy()

        # Check if any clipping occurs and reasign clipped velocities appropriately
        if (velocity_clipping_indices_upper.size != 0) and self.clip_upper_velocity == True:
            # Repeat magnitudes across all components of velocity for division
            #      [[ v_particle_1_mag, v_particle_1_mag, .... v_particle_1_mag],
            #        [v_particle_2_mag, v_particle_2_mag, .... v_particle_2_mag]...]
            # As all dimensions divided by same dimension
            # v magnitudes for each particle that needs to be clipped across all dimensions
            magnitudes_reshaped_upper = np.repeat(velocity_magnitudes_uncliped[velocity_clipping_indices_upper][:,np.newaxis],repeats=self.Ndim, axis=1)

            clipped_velocities[velocity_clipping_indices_upper] = unclipped_velocities[velocity_clipping_indices_upper]/magnitudes_reshaped_upper * self.velocity_max

        # Same thing but for lower edge
        if (velocity_clipping_indices_lower.size != 0) and self.clip_lower_velocity == True:

            magnitudes_reshaped_lower = np.repeat(velocity_magnitudes_uncliped[velocity_clipping_indices_lower][:,np.newaxis],repeats=self.Ndim, axis=1)
            clipped_velocities[velocity_clipping_indices_lower] = unclipped_velocities[velocity_clipping_indices_lower]/magnitudes_reshaped_lower * self.velocity_min


        return(clipped_velocities)

    def AffineInvariantMC_VelocityRule(self):
        """
        Implements Alg.2 from D. Foreman-Mackey et al. 2013 (arXiv:1202.3665).
        """
        velocities = np.zeros((self.NumParticles,self.Ndim))

        # loop over particles
        for particle_index in range(self.NumParticles):

            # pick a particle from the complementary ensemble
            j = np.random.choice( list(np.arange(particle_index))
                                 + list(np.arange(particle_index+1, self.NumParticles)) )
            if j>particle_index:
                Xj = self.Points[j] # use the PREVIOUS position of particle from complementary ensemble
            else:
                Xj = self.Points[j]+velocities[j] # use the NEW position of particle from complementary ensemble

            # draw random variable z ~ g(z)
            z = sample_g(a=self.affine_invariant_a)

            # Proposal point
            Y = Xj + z*(self.Points[particle_index]-Xj)

            # Check if proposed point in search bounds
            in_bounds = ( np.all(self.BoundsArray[:,0]<Y) and np.all(Y<self.BoundsArray[:,1]) )

            if in_bounds:           
                # acceptance probability
                logq = (self.Ndim-1)*np.log(z) + self.MyFunc(Y) - self.MyFunc(self.Points[particle_index])

                # decide if we accept or reject
                r = np.random.uniform()
            
                accept = (np.log(r)<logq)

                if accept:
                    velocities[particle_index] = Y - self.Points[particle_index] 
                    
        return velocities

    def ParallelAffineInvariantMC_VelocityRule(self):
        """
        Implements Alg.3 from D. Foreman-Mackey et al. 2013 (arXiv:1202.3665).
        """
        velocities = np.zeros((self.NumParticles,self.Ndim))

        Khalf = self.NumParticles//2
        indices = [np.arange(0, Khalf),
                   np.arange(Khalf+1, self.NumParticles)]

        # loop over the two halves of the swarm 
        for i in [0,1]:

            noti = 0 if i==1 else 1

            my_half_swarm = self.Points[indices[i]]
            other_half_swarm = self.Points[indices[noti]]

            offset = 0 if i==0 else Khalf

            args = [(particle_id, my_half_swarm, other_half_swarm, self.MyFunc, self.BoundsArray, velocities, self.affine_invariant_a) 
                    for particle_id in range(len(my_half_swarm))]
            # NOT FIXED YET
            V = np.array(self.Pool.starmap(ParallelStretchMove_InternalFunction, args))
            
            velocities[indices[i]] = V

        return velocities

    def ParallelAffineInvariantMC_VelocityRule_(self):
        """
        Implements the parallel Alg.3 from D. Foreman-Mackey et al. 2013 (arXiv:1202.3665)
        BUT it implements in without any parallelisation.
        """
        velocities = np.zeros((self.NumParticles,self.Ndim))

        Khalf = self.NumParticles//2
        indices = [np.arange(0, Khalf),
                   np.arange(Khalf+1, self.NumParticles)]

        # loop over the two halves of the swarm 
        for i in [0,1]:
            
            # loop over particles in this swarm half
            for particle_index in indices[i]:

                # pick a particle from the other half of the swarm
                j = np.random.choice( indices[0 if i==1 else 1] )
                Xj = self.Points[j] 

                # draw random variable z ~ g(z)
                z = sample_g(a=self.affine_invariant_a)

                # Proposal point
                Y = Xj + z*(self.Points[particle_index]-Xj)

                # Check is proposed point in search bounds
                in_bounds = ( np.all(self.BoundsArray[:,0]<Y) and np.all(Y<self.BoundsArray[:,1]) )

                if in_bounds:
                    # acceptance probability
                    logq = (self.Ndim-1)*np.log(z) + self.MyFunc(Y) - self.MyFunc(self.Points[particle_index])

                    # decide if we accept or reject
                    r = np.random.uniform()
                
                    accept = (np.log(r)<logq) 

                    if accept:
                        velocities[particle_index] = Y - self.Points[particle_index] 
                        
        return velocities

    def Hybrid_VelocityRule(self):
        """
        The PSO rule for updating the velocities including a MH MCMC part to it

        RETURN:
        ------
        clipped_velocities: numpy array
            velocities clipped to the minimum velocity for each dimension

        """
        MH_velocities = np.zeros((self.NumParticles,self.Ndim))

        for particle_index in range(self.NumParticles):
            # Draw from proposal distribution
            draw = np.random.multivariate_normal(self.Points[particle_index], self.ProposalCov, 1)[0]

            # Need to clip the draws at the boundary - wont cause any big problems for posterior distribution
            draw = np.clip(draw, self.BoundsArray[:, 0], self.BoundsArray[:, 1])

            # Note log ratio
            alpha = np.exp(self.MyFunc(draw)-self.MyFunc(self.Points[particle_index]))

            if np.random.uniform(0,1,1) <= alpha :
                MH_velocities[particle_index] = draw - self.Points[particle_index]
            else:
                MH_velocities[particle_index] = np.zeros(self.Ndim)


        best_known_swarm_point = np.tile(
            self.BestKnownSwarmPoint, self.NumParticles
        ).reshape((self.NumParticles, self.Ndim))

        unclipped_velocities = (self.Omega * self.Velocities
                                + self.PhiP * np.random.uniform(size=(self.NumParticles, self.Ndim)) * (
                                            self.BestKnownPoints - self.Points)
                                + self.PhiG * np.random.uniform(size=(self.NumParticles, self.Ndim)) * (
                                            best_known_swarm_point - self.Points) + self.MH_fraction * MH_velocities)

        # Clip velocities by the minimum velocity for each dimension to avoid pointless exploration
        #   Need to compare absolute velocities (why we have to do the np.sign business)
        clipped_velocities = np.sign(unclipped_velocities) * np.clip(np.abs(unclipped_velocities),
                                                                     a_min=self.velocity_min, a_max=None)

        return (clipped_velocities)

    def EvolveSwarm(self):
        """
        Evolve swarm through a single iteration
        """
        self.EvolutionCounter += 1

        # Update particle velocities
        self.Velocities = self.VelocityRule()

        # Update particle positions
        self.Points += self.Velocities

        # Enforce point to be within bounds
        self.EnforceBoundaries()

        # Update function values
        if self.parallel:
            self.Values = np.array( self.Pool.map(self.MyFunc, self.Points) )
        else:
            self.Values = np.array( list(map(self.MyFunc, self.Points)) )


        # Update particle's best known position
        new_best_known_values = np.maximum(self.BestKnownValues, self.Values)
        self.BestKnownPoints = np.where(np.tile(new_best_known_values==self.BestKnownValues,self.Ndim).reshape((self.Ndim,self.NumParticles)).T,
                                           self.BestKnownPoints, self.Points)
        self.BestKnownValues = new_best_known_values

        # Update swarm's best known position
        if np.max(self.Values) > self.BestKnownSwarmValue:
            self.BestKnownSwarmPoint = self.Points[np.argmax(self.Values)].copy()
            self.BestKnownSwarmValue = np.max(self.Values).copy()


        # Append spreads
        self.Spreads.append(np.ptp(self.Values))
        # Append best Swarm value to history of best swarm values
        self.FuncHistory.append(self.BestKnownSwarmValue)

    def Checkpoint(self):
        """
        Checkpoint swarm internal state
        """
        resume_file = os.path.join(self.Output, "PySO_resume.pkl")
        with open(resume_file, "wb") as f:
            pickle.dump(self, f)


    def ResumeFromCheckpoint(self):
        """
        Resume swarm from a checkpoint pickle file
        """
        resume_file = os.path.join(self.Output, "PySO_resume.pkl")
        with open(resume_file, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.clear()
        self.__dict__.update(obj.__dict__)


    def PrintStatus(self):
        """
        Print the current run status
        """
        output_str  = "Iteration: {0}, ".format(self.EvolutionCounter)
        output_str += "Max Value: {0} ".format(self.BestKnownSwarmValue)
        output_str += "at {0}, ".format(self.BestKnownSwarmPoint)
        output_str += "Spread: {0}".format(np.ptp(self.Values))
        print(output_str)

    def SaveFinalResults(self):
        """
        Save the final results to file
        """
        results_file = os.path.join(self.Output, "Results_Summary.txt")
        print("Saving final results to file {}".format(results_file))

        output_str  = "Final Results:\n"
        output_str += "Num Iterations: {0}\n".format(self.EvolutionCounter)
        output_str += "Max Value: {0}\n".format(self.BestKnownSwarmValue)
        output_str += "Best Point: {0}\n".format(self.BestKnownSwarmPoint)
        output_str += "Covariance Matrix:\n{0}\n".format(np.cov(self.Points.T))
        output_str += "Uncertainties = sqrt(diag(cov)): {0}\n".format(np.sqrt(np.diag(np.cov(self.Points.T))))

        file1 = open(results_file, "w")
        file1.write(output_str)
        file1.close()

        results_dictionary = {}
        results_dictionary['Labels'] = self.Model.names
        results_dictionary['Max Value'] = self.BestKnownSwarmValue
        results_dictionary['Best point'] = self.BestKnownSwarmPoint
        results_dictionary['Covariance matrix'] = np.cov(self.Points.T)
        results_dictionary['Uncertainties'] = np.sqrt(np.diag(np.cov(self.Points.T)))
        results_dictionary_file = os.path.join(self.Output, "Results.npz")
        np.savez(results_dictionary_file, **results_dictionary)


    def ContinueCondition_Vanilla(self):
        """
        When continue condition ceases to be satisfied the evolution stops

        Continue condition: If either the number of iterations is above the maximum number of iterations
        """
        return (self.EvolutionCounter<self.MaxIter )

    def ContinueCondition_Hybrid(self):
        """
        When continue condition ceases to be satisfied the evolution stops

        Hybrid continue condition: If either the number of iterations is above the maximum number of iterations
            or the best known swarm value has not improved in self.Convergence_testing_num_iterations, terminate
            the swarm
        """

        continue_evolution = True

        if (self.EvolutionCounter> int(self.Convergence_testing_num_iterations)):

            if (self.BestKnownSwarmValue - self.FuncHistory[-self.Convergence_testing_num_iterations]) < self.Tol:

                continue_evolution = False

        if self.EvolutionCounter>=self.MaxIter:

            continue_evolution = False

        return (continue_evolution)

    def CreateEvolutionHistoryFile(self):
        """
        Create a file to store the evolution history of the swarm
        """

        # Checks if a file already exists in the outdir file path
        history_file_path = os.path.join(self.Output, "SwarmEvolutionHistory.dat")
        assert not os.path.isfile(history_file_path), "Swarm evolution file already exists"

        # header string
        # "# particle_number, name1, name2, name3, ..., function_value\n"
        header_string = "# particle_number, "
        for name in self.Model.names:
            header_string += name + ", "
        header_string = header_string[:-2] + ", function_value\n"

        file = open(history_file_path, "w")
        file.write(header_string)
        file.close()


    def SaveSwarmEvolution(self):
        """
        At each checkpoint append the evolution of the swarm to the history file
        """
        history_file_path = os.path.join(self.Output, "SwarmEvolutionHistory.dat")

        # "# particle_number, name1, name2, name3, ..., function_value\n"
        for i in range(self.NumParticles):
            string = str(i) + ", "
            string += np.array2string(self.Points[i], separator=', ')[1:-1].replace('\n', '')
            string += ", {}\n".format(self.Values[i])
            file = open(history_file_path, "a")
            file.write(string)
            file.close()


    def PlotSwarmEvolution(self):
        """
        Various plots showing the swarm evolution.

        WARNING: Plotting the whole swarm evolution including the pairplots gets extremely
            expensive for large number of iterations and/or points.
        """
        history_file_path = os.path.join(self.Output, "SwarmEvolutionHistory.dat")
        swarm_points = np.loadtxt(history_file_path, skiprows=1, delimiter=',')

        palette = sns.color_palette("hls", self.NumParticles)
        plt.figure()

        # Function values of each swarm point
        for i in range(self.NumParticles):
            traj = np.array(swarm_points[i::self.NumParticles])
            plt.plot(self.nPeriodicCheckpoint*np.arange(len(traj)),
                     traj[:,-1],'-', marker='o', markersize=3, color=palette[i], alpha=0.5)

        plt.xlabel("Iteration")
        plt.ylabel("Function Values")
        outfile = os.path.join(self.Output, "FunctionValues.png")
        plt.savefig(outfile)
        plt.clf()


        # TODO: Don't waste time plotting duplicates ie dont plot x vs y plot after plotting y vs x
        # Trajectory for each pair of params
        for j, name_x in enumerate(self.Model.names):
            for name_y in self.Model.names[j+1:]:
                plt.figure()
                for i in range(self.NumParticles):
                    traj = np.array(swarm_points[i::self.NumParticles])
                    plt.plot(traj[:,1+self.Model.names.index(name_x)], traj[:,1+self.Model.names.index(name_y)],
                             '-', marker='o', markersize=3, color=palette[i], alpha=0.5)
                plt.xlabel(name_x)
                plt.ylabel(name_y)
                outfile = os.path.join(self.Output, "EvolutionTrajectory_{0}_{1}.png".format(name_x, name_y))
                plt.savefig(outfile)
                plt.clf()

    def Run(self, segmenting=False):
        """
        Run optimisation

        INPUTS:
        segmenting: boolean
            flag to indicate if we are transferring likelihoods
        """
        if segmenting==False: self.InitialiseSwarm()
        if self.SaveEvolution and self.EvolutionCounter==0:
            self.CreateEvolutionHistoryFile()
            self.SaveSwarmEvolution()

        while self.ContinueCondition():

            self.EvolveSwarm()

            if self.EvolutionCounter % self.nPeriodicCheckpoint == 0:

                if self.Verbose: self.PrintStatus()

                if self.Resume: self.Checkpoint()

                if self.SaveEvolution: self.SaveSwarmEvolution()

        self.SaveFinalResults()
        if self.Plotevolution : self.PlotSwarmEvolution()

    def __getstate__(self):
        """
        Get the state of the object. For mp+pool to work, the object must be picklable.

        Returns:
            dict: A dictionary containing the object's state.
                - 'log_posterior': The log posterior value.
                - 'Model': The model object.
        """
        return {'log_posterior': self.log_posterior,
                'Model': self.Model}

    def __setstate__(self, state):
        """
        Set the object's state from the state dictionary. For mp+pool to work, the object must be picklable.

        Parameters:
        state (dict): The state dictionary containing the object's state.

        Returns:
        None
        """
        self.log_posterior = state['log_posterior']
        self.Model = state['Model']

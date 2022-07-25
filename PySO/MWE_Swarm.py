import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing_on_dill import Pool
import os
import pickle
import seaborn as sns
from .Model import Model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Swarm(object):

    def __init__(self,
                 Model,
                 NumParticles,
                 Omega = 0.6,        # PSO rule parameter
                 Phip = 0.2,         # PSO rule parameter
                 Phig = 0.2,        # PSO rule parameter
                 Mh_fraction= 0.0,
                 Tol = 1.0e-3,
                 Maxiter = 1.0e6,
                 Periodic = None,
                 Initialguess = None,
                 Nthreads = 1,
                 Seed = None,
                 Nperiodiccheckpoint = 10,
                 Output = './',
                 Resume = False,
                 Verbose = False,
                 Saveevolution = False,
                 Velocity_min = None,
                 Velocity_minimum_factor = 100,
                 Proposalcov = None,
                 Initial_guess_v_factor = 3):
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
        Mh_fraction: float:
            parameter controlling proportion of velocity rule dictated by MCMC [defaults to 0.]
        Tol: float
            the tol on the function value between, this is the spread in function values below which
            we consider the optimization algoritm to have converged [defaults to 1.0e-3]
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


        (Other parameters)
        Nthreads: int [defaults to 1]
            Number of multiprocessing threads to use.
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
        Velocity_min: None or numpy array
            (absolute) minimum velocity in every dimension, defaults to 1/100th of each dimension
        Velocity_minimum_factor: int or float
            Set absolute minimum speed to 1/velocity_minimum_factor of the prior range specified in each parameter,
             Only relevant of velocity min array not provided by user [defaults to 100]
        Proposalcov: numpy array
            Covariance matrix for gaussian proposal distribution for MH part of velocity rule [defaults to identity]
        Initial_guess_v_factor: float/int
            Fiddle factor used to multiply initial guess spreads to initialise velocities [defaults to 3]"""

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

        self.initial_guess_v_factor = Initial_guess_v_factor

        self.Omega = Omega
        self.PhiP = Phip
        self.PhiG = Phig
        self.MH_fraction = Mh_fraction

        if Proposalcov is None:
            self.ProposalCov = np.identity(self.Ndim)
        else:
            self.ProposalCov = Proposalcov

        self.EvolutionCounter = 0
        self.MaxIter = Maxiter

        self.InitialGuess = Initialguess

        self.Seed = Seed

        self.nPeriodicCheckpoint = Nperiodiccheckpoint

        self.Output = Output


        self.Resume = Resume

        self.Verbose = Verbose

        self.SaveEvolution = Saveevolution

        self.Nthreads = Nthreads

        if Periodic is None:
            self.Periodic = [ 0 for i in range(self.Ndim)]
        else:
            assert (  len(Periodic)==self.Ndim  and  all(np.isin(Periodic, [0,1]))  )
            self.Periodic = list(Periodic)

        self.PeriodicParamRanges = np.array([
                                            np.inf if self.Periodic[i]==0 else np.ptp(b)
                                        for i, b in enumerate(self.Model.bounds)])


        self.BoundsArray = np.array(self.Model.bounds)

        #     NOTE: ptp by default positive
        self.velocity_min = Velocity_min
        self.velocity_minimum_factor = Velocity_minimum_factor

        if self.velocity_min is None:
            self.velocity_min = np.ptp(self.BoundsArray,axis=1)/self.velocity_minimum_factor

        if self.MH_fraction == 0:
            # The velocity rule is the PSO standard rule when no MH
            self.VelocityRule = self.PSO_VelocityRule
        else:
            self.VelocityRule = self.Hybrid_VelocityRule


        # Only used for hierarchical PSO searches
        self.Hierarchical_step = 0
        self.Spreads = []
        self.FuncHistory = []

    def MyFunc(self, p):
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
        log_posterior = self.Model.log_likelihood(par_dict)
        return log_posterior


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

            # Initialise the particle positions/velocities with random vectors
            for i in range(self.NumParticles):
                for j in range(self.Ndim):

                    Low, High = self.Model.bounds[j]
                    self.Points[i][j] = np.random.uniform(Low, High)

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

            # Initialise the particle function values
            self.Pool = Pool(self.Nthreads)
            self.Values = np.array( self.Pool.map(self.MyFunc, self.Points) )

            # Initialise each particle's best known position to initial position
            self.BestKnownPoints = self.Points.copy()
            self.BestKnownValues = self.Values.copy()

            # Update the swarm's best known position
            self.BestKnownSwarmPoint = self.BestKnownPoints[np.argmax(self.BestKnownValues)]

            self.BestKnownSwarmValue = np.max(self.BestKnownValues)

            # Calculate initial function value spread for switch condition
            self.Spreads.append(np.ptp(self.Values))

    def QuadraticWindow(self, x):
        """
        A very simple quadratic window function
        """
        return 4. * x * (1.-x)


    def EnforceBoundaries(self):
        """
        Boundary conditions on the edge of the search region
        """
        # Periodic BCs
        self.Points = self.BoundsArray[:,0] + np.fmod(self.Points-self.BoundsArray[:,0],self.PeriodicParamRanges)

        # Hard edges
        clipped_points = np.clip(self.Points, a_min=self.BoundsArray[:,0], a_max=self.BoundsArray[:,1])

        # Reflective boundary velocities
        # This does mess with the MH MCMC part of the velocity rule, so proposals near prior boundary in the case MH_fraction=1.0 will be affected
        velocity_reflection_indices = np.where(clipped_points != self.Points)

        self.Points = clipped_points

        for particle,param_index in zip(velocity_reflection_indices[0],velocity_reflection_indices[1]):
            self.Velocities[particle,param_index] *= -1

    def PSO_VelocityRule(self):
        """
        The standard PSO rule for updating the velocities

        RETURN:
        ------
        clipped_velocities: numpy array
            velocities clipped to the minimum velocity for each dimension

        """
        best_known_swarm_point = np.tile(
                              self.BestKnownSwarmPoint, self.NumParticles
                                  ).reshape((self.NumParticles, self.Ndim))
        unclipped_velocities = (self.Omega * self.Velocities
               + self.PhiP * np.random.uniform(size=(self.NumParticles,self.Ndim)) * ( self.BestKnownPoints - self.Points )
               + self.PhiG * np.random.uniform(size=(self.NumParticles,self.Ndim)) * ( best_known_swarm_point - self.Points))

        # Clip velocities by the minimum velocity for each dimension to avoid pointless exploration
        #   Need to compare absolute velocities (why we have to do the np.sign business)
        clipped_velocities = np.sign(unclipped_velocities)*np.clip(np.abs(unclipped_velocities), a_min = self.velocity_min, a_max= None)

        return (clipped_velocities)

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

        # # Update function values
        self.Values = np.array( self.Pool.map(self.MyFunc, self.Points) )


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
        with open(pickle_file, "rb") as f:
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


    def ContinueCondition(self):
        """
        When continue condition ceases to be satisfied the evolution stops
        """
        spread = np.ptp(self.Values)
        return (self.EvolutionCounter<self.MaxIter )


    def CreateEvolutionHistoryFile(self):
        """
        Create a file to store the evolution history of the swarm
        Insert header line
        """
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
        Various plots showing the swarm evolution
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
        if self.SaveEvolution: self.PlotSwarmEvolution()





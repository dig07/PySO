import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy
import dill as pickle

from .Model import Model
from PySO.Clustering_Swarms import Clustering
from .MWE_Swarm import Swarm as Swarm

try:
    print('Defaulting to torch.multiprocessing')
    from torch.multiprocessing import Pool, set_start_method
    set_start_method('spawn',force=True)
except: 
    print('PyTorch not installed, not using torch.multiprocessing, using pathos.multiprocessing instead')
    from pathos.multiprocessing import ProcessingPool as Pool

class HierarchicalSwarmHandler(object):

    def __init__(self,
                 Hierarchical_models,
                 Numswarms,
                 Numparticlesperswarm,
                 Omega = 0.6,
                 Phip = 0.2,
                 Phig = 0.2,
                 Mh_fraction = 0.0,
                 Swarm_kwargs={},
                 Output = './',
                 Minimum_velocities=None,
                 Nperiodiccheckpoint = 10,
                 Swarm_names = None,
                 Verbose = False,
                 Saveevolution = False,
                 Maxiter = 1e4,
                 Resume = True,
                 Maximum_number_of_iterations_per_step=400,
                 Minimum_exploration_iterations = 50,
                 Initial_exploration_limit= 150,
                 Clustering_indices = None,
                 Use_func_vals_in_clustering = False,
                 Kick_velocities = True,
                 Fitness_veto_fraction = 0.05,
                 Max_particles_per_swarm = None,
                 Velocity_at_segmentation = 'Transfer',
                 Clustering_min_membership = 5,
                 Clustering_max_clusters = 70,
                 Veto_function = None,
                 Tol = 1.0e-2,
                 Convergence_testing_num_iterations = 50,
                 Nthreads = None,
                 Constriction_ladder = None,
                 Constriction_kappa_ladder = None):
        """

        REQUIRED INPUTS
        ------
        Hierarchical_models: list,
            list of Hierarchical model objects
        NumSwarms: int
            Number of Swarms to initialise.
        NumParticlesPerSwarm: list of ints,
            list containing number of particles to be assigned to each swarm.


        OPTIONAL INPUTS
        ---------------
        Omega: float or list
            the omega parameter for each hierarhical model, inertial coefficient for velocity updating [defaults to .6]
        PhiP: float or list
            the phi_p parameter for each hierarhical model, cognitive coefficient for velocity updating [defaults to .2]
        PhiG: float or list
            the phi_g parameter for each hierarhical model, social coefficient for velocity updating [defaults to .2]
        Mh_fraction: float
            parameter controlling proportion of velocity rule dictated by MCMC, for each hierarchical model [defaults to 0.]
        Swarm_kwargs: dict
            dictionary of common arguments between all swarms
        Output: str
            folder in which to save output [defaults to './']
        Minimum_velocities: list, list of arrays or None
            Minimum velocities for every hierarchical segment [defaults to None]
            If None, uses the default setting for MWE swarms at each hierarchical level
        nPeriodicCheckpoint: int
            number of iterations between printing checkpoints [defaults to 10]
            (Note the swarm will checkpoint its position on every iteration, needed for sampling case)
        Swarm_names: None or list of strings
            Names for each of the swarms [defaults to ordered numbered list]
        Verbose: bool
            Verbosity [defaults to False]
        SaveEvolution: bool
            save the entire evolution of the swarm [defaults to False]
        Maxiter: int
            maximum number of iterations the ensemble will run for [defaults to 1e4]
        Resume: bool
            look for resume pkl file on startup and checkpoint during run [defaults to False]
        Maximum_number_of_iterations_per_step: int
            Maximum number of iterations per step
        Minimum_exploration_iterations: int
            Minimum number of iterations to be done in each step before stall condition evaluated [defaults to 50]
        Initial_exploration_limit: int
            Minimum number of iterations done in the very first step before stall condition evaluated [defaults to 150]
        clustering_indices: None or list of int
            Parameter position indexes to use for relabelling/clustering step [defaults to all parameters]
        use_func_vals_in_clustering: boolean
            Boolean flag for using function values for clustering or not [defaults to False]
        Kick_velocities: boolean
            Boolean flag for reinitialising velocities from position distribution
            on clustering and segmenting [defaults to True]
        fitness_veto_fraction: float
            Fraction of Best swarm position below which we throw away new swarms [defaults to 0.05]
        max_particles_per_swarm: integer
            Maximum number of particles per swarm [defaults to int(total_num_particles/10)]
        Velocity_at_segmentation: Either 'Transfer', 'Zero' or 'Redraw'.
            Sets rule for initial velocities at each segmentation. 
                If 'Transfer', transfer velocities from previous swarm to new swarm
                If 'Zero', set initial velocities to zero for new swarm
                If 'Redraw', draw new velocities from the new swarms particle positions using a normal distribution
        clustering_min_membership: int
            minimum number of particles in each swarm [defaults to 5]
        clustering_max_clusters: int
            maximum number of clusters to test for the clustering [defaults to 70]#
        Veto_function: function or NoneType
            Function to be used for vetoing swarms, generally should accept the swarms best parameters and return a boolean. 
                Vetos are generally defined as some function of the position in parameter space and the segment number. 
        Tol: float
            the minimum improvement on functionvalue for each swarm that we class as not stalled [defaults to 1e-2]
        Convergence_testing_num_iterations: int
            If best swarm value has not improved over this many last iterations (improved past Tol) [defaults to 50]
        Nthreads: int 
            Number of threads to use for parallel processing [defaults to None]
            Note: One global processor pool is used for all the swarms. This is to avoid the overhead of creating and destroying pools for each swarm.
            If None, defaults to a serial version.
        Constriction_ladder: None or list
            Constriction factor for each hierarchical model, defaults to None. If none set to False for each segment. 
        Constriction_kappa_ladder: None or list
            Constriction kappa factor for each hierarchical model, defaults to None. If none set to 1.0 for each segment. 
                Only used if Constriction_factor for that given swarm is true. 
        """
        assert len(Hierarchical_models)>1, "Please input multiple models for Hierarchical PSO search"

        self.Hierarchical_models = Hierarchical_models

        # Ladder for PSO hyper-parameters
        self.Omegas = Omega
        self.PhiPs = Phip
        self.PhiGs = Phig
        self.MH_fractions = Mh_fraction

        # If PSO hyper-parameters are only a float, replicate them for each step in the ladder
        if type(self.Omegas) == float:
            self.Omegas = [self.Omegas] * len(self.Hierarchical_models)
            self.PhiPs = [self.PhiPs] * len(self.Hierarchical_models)
            self.PhiGs = [self.PhiGs] * len(self.Hierarchical_models)
            self.MH_fractions = [self.MH_fractions] * len(self.Hierarchical_models)
        else:
            assert len(self.Omegas) == len(self.Hierarchical_models), "Please ensure your PSO parameter lists correspond to the correct number of hierarchical steps "

        # Parameter names
        self.Model_axis_names = self.Hierarchical_models[1].names

        # Number of dimensions
        self.Ndim = len(self.Model_axis_names)

        self.NumSwarms = Numswarms

        # NOTE THIS NAME IS MISLEADING, this is just the total size of the initial swarm
        self.NumParticlesPerSwarm = Numparticlesperswarm

        # Common parameters for all swarms
        self.Swarm_kwargs = Swarm_kwargs

        self.nPeriodicCheckpoint = Nperiodiccheckpoint

        self.BestKnownEnsemblePoint = np.zeros(self.Ndim)
        self.BestKnownEnsembleValue = None
        self.BestCurrentSwarm = None

        self.Verbose = Verbose

        self.SaveEvolution = Saveevolution

        # Refering to the hierarchical steps
        self.Maximum_number_of_iterations_per_step = Maximum_number_of_iterations_per_step

        # Minimum number of iterations the swarms will conduct before they evaluate the stall condition
        self.Minimum_exploration_iterations = Minimum_exploration_iterations

        # Minimum number of iterations for the first model
        self.Initial_exploration_limit = Initial_exploration_limit

        # Maximum number of iterations the whole ensemble of swarms will run for
        self.Maxiter = Maxiter

        # UNTESTED RESUME FUNCTIONALITY
        self.Resume = Resume

        # Output directory
        self.Output = Output

        # If we have given the swarm names, otherwise default to numbered list
        self.Swarm_names = Swarm_names
        if self.Swarm_names == None: self.Swarm_names = np.arange(self.NumSwarms) # Defaults to numbered list of swarms

        # If the clustering is done only in certain parameters, otherwise cluster in all dimensions (Not including objective function value)
        self.clustering_indices  = Clustering_indices
        if self.clustering_indices == None: self.clustering_indices = np.arange(self.Ndim) # Use all parameters by default in clustering

        # If the objective function value is to be used in the clustering process.
        self.use_func_vals_in_clustering = Use_func_vals_in_clustering

        # If true, draw new velocities from the new swarms particle positions using a normal distribution [ DEFAULTS TO TRUE ]
        # If false, use previous particle velocities
        self.kick_velocities = Kick_velocities

    
        self.velocity_at_segmentation = Velocity_at_segmentation

        # self.fitness_veto_fraction * best_objective_function_Val below which we throw swarms away (reassign particles to other swarms)
        self.fitness_veto_fraction = Fitness_veto_fraction

        # Maximum particles per swarm, defaults to total particles over 10
        self.max_particles_per_swarm = Max_particles_per_swarm
        if self.max_particles_per_swarm == None: self.max_particles_per_swarm = int(self.NumParticlesPerSwarm/10)

        # Veto function to be used for vetoing swarms, generally should accept the swarms best parameters and return a boolean.
        self.Veto_function = Veto_function
        
        # Constriction ladder
        if Constriction_ladder != None:
            self.Constriction_ladder = Constriction_ladder
            self.Constriction_kappa_ladder = Constriction_kappa_ladder
        else:
            self.Constriction_ladder = [False]*len(self.Hierarchical_models)
            self.Constriction_kappa_ladder = [1.0]*len(self.Hierarchical_models)



        # Minimum velocities ladder
        if np.all(Minimum_velocities) != None:

            # If minimum velocities only given for each dimension
            if len(Minimum_velocities) == self.Ndim:
                self.Minimum_velocities = [np.array(Minimum_velocities)]*len(self.Hierarchical_models)

            # If minimum velocities provided for each dimension for each hierarchical level
            elif len(Minimum_velocities) == len(self.Hierarchical_models):
                self.Minimum_velocities = Minimum_velocities

            #TODO: deal with this sensibly
            # If nothing provided then make all the minimum velocities 0
            else:
                self.Minimum_velocities = [0] * len(self.Hierarchical_models)

        # Number of threads to use for parallel processing (In one global processor pool for all the swarms)
        self.Nthreads = Nthreads


        #Make new pool for parallel computations
            # This pool will be used throughout the entire run

        if self.Nthreads != None:
            self.parallel = True
            self.Global_Pool = Pool(self.Nthreads)
        else:
            self.parallel = False


        # Initialise swarms
        self.InitialiseSwarms()

        # frozen swarms are temporarily held in this dict
        self.frozen_swarms = {}

        # Counts which hierarchical model we are on
        self.Hierarchical_model_counter = 0

        # Boolean variable to indicate if all swarms at the current level have stalled
        self.AllStalled = False

        # Variable to check if all swarms have finishing iterating on the last hierarchical model.
        self.swarm_stepping_done = False

        # Clustering parameters (See docstrings)
        self.clustering_min_membership = Clustering_min_membership
        self.clustering_max_clusters = Clustering_max_clusters

        # RESUME FUNCTIONALITY UNTESTED TODO: TEST
        resume_file = os.path.join(self.Output, "PySO_resume.pkl")

        # Stall testing parameters
        self.Tol = Tol
        self.Convergence_testing_num_iterations = Convergence_testing_num_iterations

        if self.Resume and os.path.exists(resume_file):
            print('Resuming from file {}'.format(resume_file))
            self.ResumeFromCheckpoint()

    def InitialiseSwarms(self):
        """
        Initialise the swarm points, values and velocities

        """
        # self.Swarms contains all the swarms.
        if self.parallel == False:
            self.Swarms = {self.Swarm_names[swarm_index]: Swarm(self.Hierarchical_models[0], self.NumParticlesPerSwarm,
                                                            Omega=self.Omegas[0], Phig= self.PhiGs[0], Phip=self.PhiPs[0], Mh_fraction=self.MH_fractions[0],
                                                            Velocity_min=self.Minimum_velocities[0],Nthreads=None,Constriction=self.Constriction_ladder[0],
                                                            Constriction_kappa=self.Constriction_kappa_ladder[0],**self.Swarm_kwargs)
                        for swarm_index in self.Swarm_names}

        else:
            self.Swarms = {self.Swarm_names[swarm_index]: Swarm(self.Hierarchical_models[0], self.NumParticlesPerSwarm,
                                                                Omega=self.Omegas[0], Phig= self.PhiGs[0], Phip=self.PhiPs[0], Mh_fraction=self.MH_fractions[0],
                                                                Velocity_min=self.Minimum_velocities[0],Provided_pool=self.Global_Pool,Constriction=self.Constriction_ladder[0],
                                                                Constriction_kappa=self.Constriction_kappa_ladder[0],**self.Swarm_kwargs)
                        for swarm_index in self.Swarm_names}

        initial_best_positions = []
        initial_max_func_vals = []

        #Initialise each swarm and work out the initial best position for the ensemble
        for Swarm_ in self.Swarms.values():
            Swarm_.InitialiseSwarm()
            initial_best_positions.append(Swarm_.BestKnownSwarmPoint)
            initial_max_func_vals.append(Swarm_.BestKnownSwarmValue)

        self.BestKnownEnsembleValue = np.max(initial_max_func_vals)
        self.BestKnownEnsemblePoint = initial_best_positions[np.argmax(initial_max_func_vals)]
        self.BestCurrentSwarm = list(self.Swarms.keys())[np.argmax(initial_max_func_vals)]
        self.EvolutionCounter = 0

        print('Swarm initialisation finished....')
    def EvolveSwarms(self):
        """
        Evolve every swarm through a single iteration
        """
        self.EvolutionCounter += 1

        # TODO: Could probably speed this up by assigning different pools of CPUs to different Swarms
        for name in list(self.Swarms.keys()):
            self.Swarms[name].EvolveSwarm()

            if np.max(self.Swarms[name].BestKnownSwarmValue) > self.BestKnownEnsembleValue:
                self.BestKnownEnsembleValue = np.max(self.Swarms[name].BestKnownSwarmValue)
                self.BestKnownEnsemblePoint = self.Swarms[name].Points[np.argmax(self.Swarms[name].Values)]
                self.BestCurrentSwarm = name

    def veto_and_redistribute(self):
        """

        Veto peaks due to their function values being below the threshold (fraction of peak ensemble value) and
        cap swarms for redistribution

        RETURNS:
        -------
        num_particles_redistributed: int
            Number of particles to be redistributed
        """
        num_particles_redistributed = 0

        # lowest value of any particle in the entire ensemble (used in the insignificant peak veto below)
        lowest_ensemble_val = np.min([np.min(self.frozen_swarms[swarm_index].Values) for swarm_index in list(self.frozen_swarms.keys())])

        for swarm_index in list(self.frozen_swarms.keys()):
            num_particles_in_swarm =  self.frozen_swarms[swarm_index].Points.shape[0]

            # if we actually have a veto function
            if self.Veto_function != None:
                # Need parmaeter space position position to compute SNR (coherent)
                # Need best known swarm value (Upsilon) to compute the false alarm rate 
                # Need the segment number to compute the veto
                veto = self.Veto_function(self.frozen_swarms[swarm_index].BestKnownSwarmPoint, 
                                          self.frozen_swarms[swarm_index].BestKnownSwarmValue, 
                                          self.Hierarchical_models[self.Hierarchical_model_counter].segment_number)
                if veto: 
                    print('Swarm ',swarm_index,' vetoed by veto funtion, redistributing...')
                    # Remove it from the frozen swarms, just add up how many particles need to be redistributed
                    num_particles_redistributed += self.frozen_swarms[swarm_index].Points.shape[0]
                    self.frozen_swarms.pop(swarm_index)
    

            # If not using a veto function, use the fitness veto fraction
            # Check if the peak being explored is insignificant
            elif (self.frozen_swarms[swarm_index].BestKnownSwarmValue - lowest_ensemble_val) < self.fitness_veto_fraction*(self.BestKnownEnsembleValue-lowest_ensemble_val):
                print('Swarm ',swarm_index,' below the fitness threshold, redistributing...')
                # Remove it from the frozen swarms, just add up how many particles need to be redistributed
                num_particles_redistributed += self.frozen_swarms[swarm_index].Points.shape[0]
                self.frozen_swarms.pop(swarm_index)

            elif num_particles_in_swarm > self.max_particles_per_swarm:

                # Add up how many particles need to be redistributed
                num_particles_redistributed += int(self.frozen_swarms[swarm_index].Points.shape[0] - self.max_particles_per_swarm)

                # Find the lowest fitness particles
                lowest_fitness_particle_indices = np.argsort(self.frozen_swarms[swarm_index].Values)[:(self.frozen_swarms[swarm_index].Points.shape[0] -
                                                                                                       self.max_particles_per_swarm)]

                # Remove those particles from the frozen swarms datastructure
                self.frozen_swarms[swarm_index].Points = np.delete(self.frozen_swarms[swarm_index].Points,lowest_fitness_particle_indices,0)
                self.frozen_swarms[swarm_index].Velocities = np.delete(self.frozen_swarms[swarm_index].Velocities,lowest_fitness_particle_indices,0)
                self.frozen_swarms[swarm_index].Values = np.delete(self.frozen_swarms[swarm_index].Values,lowest_fitness_particle_indices,0)
                self.frozen_swarms[swarm_index].BestKnownSwarmPoints = np.delete(self.frozen_swarms[swarm_index].BestKnownPoints,lowest_fitness_particle_indices,0)
                self.frozen_swarms[swarm_index].BestKnownSwarmValue = np.delete(self.frozen_swarms[swarm_index].BestKnownValues,lowest_fitness_particle_indices,0)

                print('Swarm ',swarm_index, ' is over the maximum size per swarm, redistributing ',num_particles_in_swarm -
                      self.max_particles_per_swarm,' Particles')

        return(num_particles_redistributed)



    def Reallocate_particles(self):
        """Use all particles in current swarms, cluster them based on features and reallocate"""

        # Dont want to converge to or explore peaks that are below a certain threshold compared to the best of the entire ensemble
        # But dont want this redistribution to take place on the first "exploratory" swarm
        # Also dont want to redistribute if we are mostly doing MH MCMC velocity rule, as we dont expect strong clustering there
        if self.Hierarchical_model_counter != 0 and self.MH_fractions[self.Hierarchical_model_counter]< 0.1:
            # this is the TOTAL number of particles to be redistributed to a new swarm
            num_particles_redistributed = self.veto_and_redistribute()

        # Extract positions to be used for clustering, only using the indices of parameters that are well measured AND function value     #
        # Feature_array - Particle positions, function values, not all components (Specially not ones well measured)

        clustering_parameter_positions = np.array([np.take(self.frozen_swarms[swarm_index].Points,self.clustering_indices,axis=1) for swarm_index
                               in self.frozen_swarms.keys()],dtype=object)

        clustering_parameter_positions = np.concatenate(clustering_parameter_positions)


        if self.use_func_vals_in_clustering == True:
            clustering_function_values = np.array([self.frozen_swarms[swarm_index].Values for swarm_index
                                                   in self.frozen_swarms.keys()])
            clustering_function_values = np.concatenate(clustering_function_values)
            clustering_features = np.column_stack((clustering_parameter_positions, clustering_function_values))
        else:
            clustering_features = clustering_parameter_positions

        # min membership is the minimum number of particles per swarm and max clusters is the maximum number of clusters
        K, memberships = Clustering(clustering_features,min_membership=self.clustering_min_membership,max_clusters=self.clustering_max_clusters)


        total_particle_positions = np.vstack([self.frozen_swarms[swarm_index].Points for swarm_index
                               in self.frozen_swarms.keys()])
        total_particle_velocities = np.vstack([self.frozen_swarms[swarm_index].Velocities for swarm_index
                               in self.frozen_swarms.keys()])

        print('Reinitiating swarms with Omega: ',self.Omegas[self.Hierarchical_model_counter+1],
              ' PhiP: ',self.PhiPs[self.Hierarchical_model_counter+1],
              ' PhiG: ',self.PhiGs[self.Hierarchical_model_counter+1])

        # Create each swarm
        for swarm_index in range(K):

              swarm_particle_positions = total_particle_positions[np.where(memberships == swarm_index)[0]]
              swarm_particle_velocities = total_particle_velocities[np.where(memberships == swarm_index)[0]]

              # Note swarm velocities are not carried over to next segment if self.velocity_rule_at_segmentation is 'Redraw' or 'Zero'
              self.Swarms[swarm_index] = self.Reinitiate_swarm(swarm_particle_positions, swarm_particle_velocities)

              # Force all swarms to use the same global pool
              if self.parallel == True:
                self.Swarms[swarm_index].Pool = self.Global_Pool 
                

        # Check to make sure that we arent on the first segment and there are actually particles to be redistributed (from veto)


        if self.Hierarchical_model_counter != 0 and num_particles_redistributed>0:
            # Redistribute particles into the best swarm we currently are tracking
            #       Do this by placed our "redistributed swarm" on top of the best swarm

             # Find the best swarm
            best_swarm_index = np.argmax([self.Swarms[swarm_index].BestKnownSwarmValue for swarm_index in list(self.Swarms.keys())])

            # Its particle positions
            parameter_positions = self.Swarms[best_swarm_index].Points

            # Distribute the new swarms positions basically using the centre point of all the current swarms (This might not be a good idea in the end)
            cov = np.cov(parameter_positions.T)/2
            position_mean = np.mean(parameter_positions,axis=0)
            velocity_mean = np.zeros(self.Ndim)

            redistributed_particle_positions = np.random.multivariate_normal(position_mean, cov, size=num_particles_redistributed)
            redistributed_particle_velocities = np.random.multivariate_normal(velocity_mean, cov, size=num_particles_redistributed)

            # Extra redistributed swarm
            # Note swarm velocities are not carried over to next segment if self.velocity_rule_at_segmentation is 'Redraw' or 'Zero'
            self.Swarms[K] = self.Reinitiate_swarm(redistributed_particle_positions, redistributed_particle_velocities)

            # Force all swarms to use the same global pool
            if self.parallel == True:
                self.Swarms[swarm_index].Pool = self.Global_Pool 
            

            
        # Empty the frozen swarms dict as we are done with the old swarms
        self.frozen_swarms = {}
        self.AllStalled = False

        self.Hierarchical_model_counter += 1

    def Reinitiate_swarm(self,positions,velocities,
                         Omega=None,
                         PhiP=None,
                         PhiG=None):
        """
        Reinitiate swarm given some positions and velocities.

        INPUTS:
        ------
        positions: array (number of particles, self.Ndim)
            initial positions for new swarm
        velocities: array (number of particles, self.Ndim)
            initial velocities for new swarm particles
                Only use this for new velocities if self.velocity_rule_at_segmentation is 'Transfer'

        OPTIONAL INPUTS:
        ------
        Omega: float
            Inertia parameter for new swarm [defaults to the hierarchical model list]
        PhiP: float
            Personal cognitive parameter for new swarm [defaults to the hierarchical model list]
        PhiG: float
            Group parameter for new swarm [defaults to the hierarchical model list]


        RETURNS:
        ------
        newswarm: swarm object
            new swarm initiated
        """
        if (Omega is None) and (PhiP is None) and (PhiG is None):
            Omega = self.Omegas[self.Hierarchical_model_counter + 1]
            PhiP = self.PhiPs[self.Hierarchical_model_counter + 1]
            PhiG = self.PhiGs[self.Hierarchical_model_counter + 1]
            MH_fraction = self.MH_fractions[self.Hierarchical_model_counter + 1]
            Velocity_min = self.Minimum_velocities[self.Hierarchical_model_counter + 1]


        num_particles = positions.shape[0]

        if self.parallel == True: 
            newswarm = Swarm(self.Hierarchical_models[self.Hierarchical_model_counter + 1],num_particles,
                            Omega=Omega, Phip=PhiP, Phig=PhiG, Mh_fraction=MH_fraction ,Velocity_min=Velocity_min,Provided_pool=self.Global_Pool,
                            Constriction=self.Constriction_ladder[self.Hierarchical_model_counter + 1],Constriction_kappa=self.Constriction_kappa_ladder[self.Hierarchical_model_counter + 1],
                            **self.Swarm_kwargs)
        else:
            newswarm = Swarm(self.Hierarchical_models[self.Hierarchical_model_counter + 1],num_particles,
                Omega=Omega, Phip=PhiP, Phig=PhiG, Mh_fraction=MH_fraction ,Velocity_min=Velocity_min,Nthreads=None,
                Constriction=self.Constriction_ladder[self.Hierarchical_model_counter + 1],Constriction_kappa=self.Constriction_kappa_ladder[self.Hierarchical_model_counter + 1],
                **self.Swarm_kwargs)

        newswarm.EvolutionCounter = 0

        # New points clipped at the boundaries
        newswarm.Points = np.clip(positions,a_min=np.array(self.Hierarchical_models[self.Hierarchical_model_counter + 1].bounds)[:,0],
                                  a_max=np.array(self.Hierarchical_models[self.Hierarchical_model_counter + 1].bounds)[:,1])

        if self.kick_velocities == True:
            # Kick the reinitialised velocities
            # Regenerate velocities from a normal distribution specified by the covariance of particles in swarm
            vel_cov = np.cov(positions.T)

            # Reinitialising velocities with a mean of zero (Shape of mean is 1D with length number of dimensions)
            mean = np.zeros(positions.shape[1])

            # Draw velocities from normal distribution specified by position
            newswarm.Velocities = np.random.multivariate_normal(mean, vel_cov, size=num_particles)
        else:
            newswarm.Velocities = velocities


        if self.velocity_at_segmentation == 'Redraw':

            # Work out the peak to peak of the positions in each axis.
            ptp_vel_bounds = np.ptp(np.array([np.min(positions,axis=0),np.max(positions,axis=0)]).T,axis=1)

            # draw velocities from U[-ptp/2,ptp/2] for each axis
            newswarm.Velocities = (ptp_vel_bounds) * np.random.random_sample(size=(num_particles,self.Ndim)) - ptp_vel_bounds/2

        elif self.velocity_at_segmentation == 'Zero':

            # Set all velocities to zero at new swarm 
            newswarm.Velocities = np.zeros((num_particles,self.Ndim))
        elif self.velocity_at_segmentation == 'Transfer':
            # Transfer velocities from previous swarm to new swarm (already done above)
            pass
        # Carry over points from previous models optimization
        newswarm.BestKnownPoints = copy.deepcopy(newswarm.Points)

        # Recalculate best personal known values:
        for i in range(newswarm.NumParticles):
            # TODO: This be paralellized
            newswarm.BestKnownValues[i] = newswarm.Model.log_likelihood(
                dict(zip(newswarm.Model.names, newswarm.Points[i])))

        # First values for each particle are by definition their best known values
        newswarm.Values = copy.deepcopy(newswarm.BestKnownValues)

        # Best point and swarm value
        newswarm.BestKnownSwarmPoint = newswarm.BestKnownPoints[np.argmax(newswarm.BestKnownValues)]
        newswarm.BestKnownSwarmValue = np.max(newswarm.BestKnownValues)

        return (newswarm)

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
        output_str  = "\n Iteration: {0} \n".format(self.EvolutionCounter)
        for swarm_name in list(self.Swarms.keys()):
            output_str += "\n"
            output_str += "Swarm: {0}, ".format(swarm_name)
            output_str += "Max value: {0}, ".format(self.Swarms[swarm_name].BestKnownSwarmValue)
            output_str += "at {0}, ".format(self.Swarms[swarm_name].BestKnownSwarmPoint)
            output_str += "with spread {0}".format(self.Swarms[swarm_name].Spreads[-1])
        print(output_str)


    def CreateEvolutionHistoryFile(self):
        """
        Create a file to store the evolution history of the swarm
        Insert header line
        """
        history_file_path = os.path.join(self.Output, "EnsembleEvolutionHistory.dat")

        assert not os.path.isfile(history_file_path), "Ensemble evolution file already exists"

        # header string
        # "swarm_number, particle_number, name1, name2, name3, ..., function_value\n"
        header_string = "swarm_number,particle_number,"
        for name in self.Model_axis_names:
            header_string += name + ","
        header_string = header_string + "function_value,HierarchicalModelNumber,IterationNumber\n"

        file = open(history_file_path, "w")
        file.write(header_string)
        file.close()

    def SaveEnsembleEvolution(self):
        """
        At each checkpoint append the evolution of the swarm to the history file
        """
        history_file_path = os.path.join(self.Output, "EnsembleEvolutionHistory.dat")

        # "# swarm name, particle_number, name1, name2, name3, ..., function_value\n"
        for swarm_name in list(self.Swarms.keys()):
            for particle_index in range(self.Swarms[swarm_name].NumParticles):
                string = str(swarm_name) + ","
                string += str(particle_index) + ","
                string += np.array2string(self.Swarms[swarm_name].Points[particle_index], separator=',')[1:-1].replace('\n', '')
                string += ",{}".format(self.Swarms[swarm_name].Values[particle_index])
                string += ",{}".format(self.Hierarchical_model_counter)
                string += ",{}\n".format(self.EvolutionCounter)
                file = open(history_file_path, "a")
                file.write(string)
                file.close()

        # All the frozen swarm data : Might be redundant to store these instead of with a 'frozen' flag but helps with visualisation for now
        for swarm_name in list(self.frozen_swarms.keys()):
            for particle_index in range(self.frozen_swarms[swarm_name].NumParticles):
                string = str(swarm_name) + ","
                string += str(particle_index) + ","
                string += np.array2string(self.frozen_swarms[swarm_name].Points[particle_index], separator=',')[1:-1].replace('\n', '')
                string += ",{}".format(self.frozen_swarms[swarm_name].Values[particle_index])
                string += ",{}".format(self.Hierarchical_model_counter)
                string += ",{}\n".format(self.EvolutionCounter)
                file = open(history_file_path, "a")
                file.write(string)
                file.close()
    def SaveFinalResults(self):
        """
        Save the final results to file
        """
        final_swarm_positions = np.concatenate([self.frozen_swarms[swarm_index].Points for swarm_index in list(self.frozen_swarms.keys())])
        final_swarm_values = np.hstack([self.frozen_swarms[swarm_index].Values for swarm_index in list(self.frozen_swarms.keys())])
        final_swarm_positions_filename = os.path.join(self.Output, "final_swarm_positions.txt")
        final_swarm_values_filename = os.path.join(self.Output, "final_swarm_values.txt")

        np.savetxt(final_swarm_positions_filename,final_swarm_positions)
        np.savetxt(final_swarm_values_filename,final_swarm_values)

        final_swarm_pickle_filename= os.path.join(self.Output, "final_swarm_pickle.pkl")
        # Dump final swarms into a pickle file
        pickle.dump(self.frozen_swarms.copy(), open(final_swarm_pickle_filename, "wb"))

        pass


    def ContinueCondition(self):
        """
        When continue condition ceases to be satisfied the evolution stops
        """
        #-1 since self.EvolutionCounter starts at 0
        return( self.EvolutionCounter<self.Maxiter-1)


    def check_hierarchical_step(self):
        """
        Checks if any of the swarms meet the condition to switch to the next model.

        This method iterates over the swarms and checks if any of them have reached the stall condition. If a swarm has
        stalled, it freezes the swarm and removes it from the active swarms. If all swarms have stalled, it either finishes
        the process if it's the last model or switches to the next segment.

        Returns:
            None
        """

        # If all swarms are not stalled yet
        if self.AllStalled == False:

            for swarm_index, Swarm_ in zip(list(self.Swarms.keys()),list(self.Swarms.values())):

                # If the mean of the spreads computed across the last 10 iterations has not gotten lower,
                #   Assume the swarm has stalled and thus conduct a hierarchical step

                if Swarm_.EvolutionCounter > self.Minimum_exploration_iterations and self.EvolutionCounter > self.Initial_exploration_limit:

                    if self.stall_condition(Swarm_):

                        print('\n Swarm ',str(swarm_index),' reached stall condition, freezing')

                        # Freeze until all swarms have been stalled in this given segment likelihood

                        self.frozen_swarms[swarm_index] = Swarm_

                        self.Swarms.pop(swarm_index)

                        # If all the swarms have stalled:
                        if len(list(self.Swarms.values())) == 0: self.AllStalled = True

            if self.AllStalled:

                if self.Hierarchical_model_counter+1 == len(self.Hierarchical_models):
                    print('\n All swarms stalled on the last model, finishing up!')
                    self.swarm_stepping_done = True
                    # for swarm in self.Swarms:
                    #     swarm.Pool.close()
                    #     swarm.Pool.join()
                    if self.parallel:
                        self.Global_Pool.close()
                        self.Global_Pool.join()
                    

                else:
                    print('\n All swarms stalled! Switching segments from ', str(self.Hierarchical_models[self.Hierarchical_model_counter].segment_number),
                          ' to ', str(self.Hierarchical_models[self.Hierarchical_model_counter+1].segment_number))
                    self.Reallocate_particles()


    def stall_condition(self,Swarm):
        """
        Evaluate stall condition for a given swarm.

        Current stall condition is the best swarm value has not increased more than a tolerance (default of 0.01)
        in the last some number of iterations (defaults to 50). Swarm also stalls if the evolution counter goes over
        the maximum number of iterations per step.
        """
        stalled = ((np.abs(Swarm.BestKnownSwarmValue - Swarm.FuncHistory[-self.Convergence_testing_num_iterations]) < self.Tol) or
                    (Swarm.EvolutionCounter >= self.Maximum_number_of_iterations_per_step))

        return stalled


    def Run(self):
        """
        Run optimisation/sampling for all swarms

        This method runs the optimization or sampling process for all swarms in the hierarchical swarm handler.
        It iteratively evolves the swarms until the stopping condition is met or the swarm stepping is done.
        It also handles saving evolution history and periodic checkpoints based on the specified parameters.

        Returns:
            None
        """
        if self.SaveEvolution and self.EvolutionCounter == 0:
            self.CreateEvolutionHistoryFile()
            self.SaveEnsembleEvolution()

        while self.ContinueCondition() and (self.swarm_stepping_done == False):
            self.EvolveSwarms()

            if self.EvolutionCounter % self.nPeriodicCheckpoint == 0:
                if self.Verbose:
                    self.PrintStatus()

                if self.SaveEvolution:
                    self.SaveEnsembleEvolution()

            self.check_hierarchical_step()

        self.SaveFinalResults()

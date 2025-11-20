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
                 Fitness_veto_fraction = 0.00,
                 Max_particles_per_swarm = None,
                 Velocity_at_segmentation = 'Transfer',
                 Clustering_min_membership = 5,
                 Clustering_max_clusters = 70,
                 Tol = 1.0e-2,
                 Convergence_testing_num_iterations = 50,
                 Nthreads = None,
                 batched=False):
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
        Tol: float
            the minimum improvement on functionvalue for each swarm that we class as not stalled [defaults to 1e-2]
        Convergence_testing_num_iterations: int
            If best swarm value has not improved over this many last iterations (improved past Tol) [defaults to 50]
        Nthreads: int 
            Number of threads to use for parallel processing [defaults to None]
            Note: One global processor pool is used for all the swarms. This is to avoid the overhead of creating and destroying pools for each swarm.
            If None, defaults to a serial version.
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

        self.batched = batched
    
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

        # Initialize benchmarking dictionaries
        self.evolution_timings = {
            'position_extraction': 0.0,
            'batch_function_eval': 0.0,
            'value_assignment': 0.0,
            'best_tracking': 0.0,
            'serial_evolution': 0.0,
            'total': 0.0
        }
        self.evolution_call_count = 0
        
        # Hierarchical step benchmarking
        self.hierarchical_step_timings = {
            'stall_checking': 0.0,
            'swarm_freezing': 0.0,
            'reallocation': 0.0,
            'total': 0.0
        }
        self.hierarchical_step_call_count = 0
        self.reallocation_count = 0

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
                                                            Velocity_min=self.Minimum_velocities[0],Nthreads=None,**self.Swarm_kwargs)
                        for swarm_index in self.Swarm_names}

        else:
            self.Swarms = {self.Swarm_names[swarm_index]: Swarm(self.Hierarchical_models[0], self.NumParticlesPerSwarm,
                                                                Omega=self.Omegas[0], Phig= self.PhiGs[0], Phip=self.PhiPs[0], Mh_fraction=self.MH_fractions[0],
                                                                Velocity_min=self.Minimum_velocities[0],Provided_pool=self.Global_Pool,**self.Swarm_kwargs)
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
        import time
        t_iter_start = time.time()
        
        self.EvolutionCounter += 1
        self.evolution_call_count += 1

        # BATCHED
        if self.batched and self.NumSwarms>1:
            # Extract all positions if using batched function evaluations
            t0 = time.time()
            batched_locations = []

            # Number of particles in each swarm. 
            num_particles = []
            for name in list(self.Swarms.keys()):
                batched_locations.extend(self.Swarms[name].EvolveSwarm_Hierarchical_batched_return_positions())
                num_particles.append(self.Swarms[name].NumParticles)
            self.evolution_timings['position_extraction'] += time.time() - t0

            # Batch compute all function values
            t0 = time.time()
            all_function_values = np.array(self.Swarms[name].MyFunc_batched(np.array(batched_locations)))
            # Cumulative sum to get indices for each swarm
            indices_for_swarms = np.cumsum(num_particles)
            self.evolution_timings['batch_function_eval'] += time.time() - t0

            # Assign function values back to each swarm and evolve rest of it
            t0 = time.time()
            for i, name in enumerate(self.Swarms.keys()):
                if i == 0:
                    swarm_function_values = all_function_values[:indices_for_swarms[i]]
                else:
                    swarm_function_values = all_function_values[indices_for_swarms[i-1]:indices_for_swarms[i]]

                self.Swarms[name].EvolveSwarm_Hierarchical_batched_assign_values(swarm_function_values)
            self.evolution_timings['value_assignment'] += time.time() - t0

            # Track best values
            t0 = time.time()
            for name in self.Swarms.keys():
                if np.max(self.Swarms[name].BestKnownSwarmValue) > self.BestKnownEnsembleValue:
                    self.BestKnownEnsembleValue = np.max(self.Swarms[name].BestKnownSwarmValue)
                    self.BestKnownEnsemblePoint = self.Swarms[name].Points[np.argmax(self.Swarms[name].Values)]
                    self.BestCurrentSwarm = name
            self.evolution_timings['best_tracking'] += time.time() - t0

        # NON BATCHED
        else:
            t0 = time.time()
            for name in list(self.Swarms.keys()):
                self.Swarms[name].EvolveSwarm()

                if np.max(self.Swarms[name].BestKnownSwarmValue) > self.BestKnownEnsembleValue:
                    self.BestKnownEnsembleValue = np.max(self.Swarms[name].BestKnownSwarmValue)
                    self.BestKnownEnsemblePoint = self.Swarms[name].Points[np.argmax(self.Swarms[name].Values)]
                    self.BestCurrentSwarm = name
            self.evolution_timings['serial_evolution'] += time.time() - t0
        
        self.evolution_timings['total'] += time.time() - t_iter_start

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

            # Use the fitness veto fraction
            # Check if the peak being explored is insignificant
            if (self.frozen_swarms[swarm_index].BestKnownSwarmValue - 
                  lowest_ensemble_val) < self.fitness_veto_fraction*(self.BestKnownEnsembleValue-lowest_ensemble_val):
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
        import time
        
        # Initialize timing dictionary
        timings = {}
        t_start = time.time()

        # Veto and redistribute step
        t0 = time.time()
        # Dont want to converge to or explore peaks that are below a certain threshold compared to the best of the entire ensemble
        # But dont want this redistribution to take place on the first "exploratory" swarm
        # Also dont want to redistribute if we are mostly doing MH MCMC velocity rule, as we dont expect strong clustering there
        if self.Hierarchical_model_counter != 0 and self.fitness_veto_fraction>0:
            # this is the TOTAL number of particles to be redistributed to a new swarm
            num_particles_redistributed = self.veto_and_redistribute()
        else:
            num_particles_redistributed = 0
        timings['veto_and_redistribute'] = time.time() - t0
    
        # Array pre-allocation step
        t0 = time.time()
        # Pre-allocate arrays to avoid repeated concatenation
        swarm_keys = list(self.frozen_swarms.keys())
        n_swarms = len(swarm_keys)
        
        # Get total particles count
        total_particles = sum(self.frozen_swarms[k].Points.shape[0] for k in swarm_keys)
        
        # Pre-allocate arrays
        total_particle_positions = np.empty((total_particles, self.Ndim))
        total_particle_velocities = np.empty((total_particles, self.Ndim))
        clustering_features_array = np.empty((total_particles, len(self.clustering_indices) + 
                                            (1 if self.use_func_vals_in_clustering else 0)))
        timings['array_allocation'] = time.time() - t0
        
        # Feature extraction step
        t0 = time.time()
        # Fill arrays in one pass
        idx = 0
        for swarm_key in swarm_keys:
            swarm = self.frozen_swarms[swarm_key]
            n_particles = swarm.Points.shape[0]
            
            total_particle_positions[idx:idx+n_particles] = swarm.Points
            total_particle_velocities[idx:idx+n_particles] = swarm.Velocities
            
            # Extract clustering features
            clustering_features_array[idx:idx+n_particles, :len(self.clustering_indices)] = \
                swarm.Points[:, self.clustering_indices]
            
            if self.use_func_vals_in_clustering:
                clustering_features_array[idx:idx+n_particles, -1] = swarm.Values
            
            idx += n_particles
        timings['feature_extraction'] = time.time() - t0
        
        # Clustering step
        t0 = time.time()
        # min membership is the minimum number of particles per swarm and max clusters is the maximum number of clusters
        K, memberships = Clustering(clustering_features_array,min_membership=self.clustering_min_membership,max_clusters=self.clustering_max_clusters)
        timings['clustering'] = time.time() - t0

        # Legacy vstack operations (keeping for compatibility)
        t0 = time.time()
        total_particle_positions = np.vstack([self.frozen_swarms[swarm_index].Points for swarm_index
                               in self.frozen_swarms.keys()])
        total_particle_velocities = np.vstack([self.frozen_swarms[swarm_index].Velocities for swarm_index
                               in self.frozen_swarms.keys()])
        timings['legacy_vstack'] = time.time() - t0

        print('Reinitiating swarms with Omega: ',self.Omegas[self.Hierarchical_model_counter+1],
              ' PhiP: ',self.PhiPs[self.Hierarchical_model_counter+1],
              ' PhiG: ',self.PhiGs[self.Hierarchical_model_counter+1])

        # Swarm creation step
        t0 = time.time()
        # Create swarms efficiently
        new_swarms = {}
        for swarm_index in range(K):
            mask = memberships == swarm_index
            new_swarms[swarm_index] = self.Reinitiate_swarm(
                total_particle_positions[mask],
                total_particle_velocities[mask]
            )
            if self.parallel:
                new_swarms[swarm_index].Pool = self.Global_Pool
        
        self.Swarms = new_swarms
        timings['swarm_creation'] = time.time() - t0

        # Redistribution step
        t0 = time.time()
        # Check to make sure that we arent on the first segment and there are actually particles to be redistributed (from veto)
        if self.Hierarchical_model_counter != 0 and num_particles_redistributed>0:
            # Redistribute particles into the best swarm we currently are tracking
            #       Do this by placed our "redistributed swarm" on top of the best swarm

             # Find the best swarm
            best_swarm_index = np.argmax([self.Swarms[swarm_index].BestKnownSwarmValue for swarm_index in list(self.Swarms.keys())])

            # Its particle positions
            parameter_positions = self.Swarms[best_swarm_index].Points

            # Distribute the new swarms positions basically using the centre point of all the best swarm (This might not be a good idea in the end)
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
        timings['redistribution'] = time.time() - t0
            
        # Cleanup step
        t0 = time.time()
        # Empty the frozen swarms dict as we are done with the old swarms
        self.frozen_swarms = {}
        self.AllStalled = False

        self.Hierarchical_model_counter += 1
        timings['cleanup'] = time.time() - t0
        
        # Total time
        timings['total'] = time.time() - t_start
        
        # Print comprehensive benchmark results
        print("\n" + "="*70)
        print("REALLOCATION PERFORMANCE BENCHMARK")
        print("="*70)
        print(f"Total particles processed: {total_particles}")
        print(f"Number of clusters found: {K}")
        print(f"Particles redistributed: {num_particles_redistributed}")
        print("-"*70)
        
        for step, duration in timings.items():
            if step != 'total':
                percentage = (duration / timings['total']) * 100
                print(f"{step:.<30s} {duration:>8.4f}s ({percentage:>5.1f}%)")
        
        print("-"*70)
        print(f"{'TOTAL TIME':.<30s} {timings['total']:>8.4f}s (100.0%)")
        print("="*70 + "\n")
        
        # Optional: Add warning if clustering is taking too long
        if timings['clustering'] > 5.0:
            print(f"⚠️  WARNING: Clustering took {timings['clustering']:.2f}s - consider reducing max_clusters or using larger min_membership")
        
        if timings['swarm_creation'] > timings['clustering']:
            print(f"⚠️  NOTE: Swarm creation ({timings['swarm_creation']:.2f}s) is slower than clustering - consider batching objective function evaluations")

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
                            **self.Swarm_kwargs)
        else:
            newswarm = Swarm(self.Hierarchical_models[self.Hierarchical_model_counter + 1],num_particles,
                Omega=Omega, Phip=PhiP, Phig=PhiG, Mh_fraction=MH_fraction ,Velocity_min=Velocity_min,Nthreads=None,
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


        # TODO: Pull this outside and use the global batched function evaluation if allowed. 
        # Checking for batching
        if self.Swarm_kwargs.get('batch_optimal_func') is not None and self.Swarm_kwargs.get('batch_optimal_func') is False:

            for i in range(newswarm.NumParticles):
                # TODO: This be paralellized
                newswarm.BestKnownValues[i] = newswarm.Model.objective_function(
                    dict(zip(newswarm.Model.names, newswarm.Points[i])))
        # Batched computation (at swarm level)
        elif self.Swarm_kwargs.get('batch_optimal_func') is True:
            newswarm.BestKnownValues = np.array(newswarm.MyFunc(newswarm.Points))

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

    def PrintEvolutionBenchmarks(self):
        """
        Print comprehensive benchmarking results for EvolveSwarms and check_hierarchical_step functions
        """
        if self.evolution_call_count == 0:
            print("No evolution timing data available.")
            return
        
        print("\n" + "="*80)
        print("EVOLVE SWARMS PERFORMANCE BENCHMARK")
        print("="*80)
        print(f"Total evolution iterations: {self.evolution_call_count}")
        print(f"Mode: {'BATCHED' if (self.batched and self.NumSwarms > 1) else 'SERIAL'}")
        print("-"*80)
        
        # Calculate per-iteration averages
        print(f"{'Component':<35s} {'Total (s)':>12s} {'Per-Iter (ms)':>15s} {'% of Total':>12s}")
        print("-"*80)
        
        for component, total_time in self.evolution_timings.items():
            if component == 'total':
                continue
            
            # Skip components with zero time (not used in current mode)
            if total_time < 0.0001:
                continue
                
            per_iter_ms = (total_time / self.evolution_call_count) * 1000
            percentage = (total_time / self.evolution_timings['total']) * 100 if self.evolution_timings['total'] > 0 else 0
            
            component_name = component.replace('_', ' ').title()
            print(f"{component_name:<35s} {total_time:>12.4f} {per_iter_ms:>15.2f} {percentage:>11.1f}%")
        
        print("-"*80)
        total_time = self.evolution_timings['total']
        avg_per_iter_ms = (total_time / self.evolution_call_count) * 1000
        print(f"{'TOTAL':<35s} {total_time:>12.4f} {avg_per_iter_ms:>15.2f} {'100.0':>11s}%")
        print("="*80)
        
        # Performance insights
        if self.batched and self.NumSwarms > 1:
            batch_time = self.evolution_timings['batch_function_eval']
            batch_pct = (batch_time / total_time) * 100 if total_time > 0 else 0
            print(f"\n💡 INSIGHTS:")
            print(f"   • Batch function evaluation: {batch_pct:.1f}% of total time")
            if batch_pct > 70:
                print(f"   ⚠️  Function evaluation dominates - consider optimizing objective function")
            
            assignment_time = self.evolution_timings['value_assignment']
            assignment_pct = (assignment_time / total_time) * 100 if total_time > 0 else 0
            if assignment_pct > 20:
                print(f"   ⚠️  Value assignment ({assignment_pct:.1f}%) seems high - potential overhead")
        else:
            serial_time = self.evolution_timings['serial_evolution']
            print(f"\n💡 INSIGHTS:")
            print(f"   • Running in SERIAL mode")
            if self.NumSwarms > 1:
                print(f"   💡 Consider enabling batched mode for {self.NumSwarms} swarms")
        
        iterations_per_sec = self.evolution_call_count / total_time if total_time > 0 else 0
        print(f"   • Average throughput: {iterations_per_sec:.2f} iterations/second")
        print("="*80 + "\n")
        
        # Print hierarchical step benchmarks
        if self.hierarchical_step_call_count > 0:
            print("\n" + "="*80)
            print("HIERARCHICAL STEP PERFORMANCE BENCHMARK")
            print("="*80)
            print(f"Total check_hierarchical_step calls: {self.hierarchical_step_call_count}")
            print(f"Number of reallocations: {self.reallocation_count}")
            print("-"*80)
            
            print(f"{'Component':<35s} {'Total (s)':>12s} {'Per-Call (ms)':>15s} {'% of Total':>12s}")
            print("-"*80)
            
            for component, total_time in self.hierarchical_step_timings.items():
                if component == 'total':
                    continue
                
                # Skip components with zero time
                if total_time < 0.0001:
                    continue
                    
                per_call_ms = (total_time / self.hierarchical_step_call_count) * 1000
                percentage = (total_time / self.hierarchical_step_timings['total']) * 100 if self.hierarchical_step_timings['total'] > 0 else 0
                
                component_name = component.replace('_', ' ').title()
                print(f"{component_name:<35s} {total_time:>12.4f} {per_call_ms:>15.2f} {percentage:>11.1f}%")
            
            print("-"*80)
            hier_total_time = self.hierarchical_step_timings['total']
            avg_per_call_ms = (hier_total_time / self.hierarchical_step_call_count) * 1000
            print(f"{'TOTAL':<35s} {hier_total_time:>12.4f} {avg_per_call_ms:>15.2f} {'100.0':>11s}%")
            print("="*80)
            
            # Hierarchical step insights
            if hier_total_time > 0:
                realloc_time = self.hierarchical_step_timings['reallocation']
                realloc_pct = (realloc_time / hier_total_time) * 100
                
                print(f"\n💡 INSIGHTS:")
                if self.reallocation_count > 0:
                    avg_realloc_time = realloc_time / self.reallocation_count
                    print(f"   • Average reallocation time: {avg_realloc_time:.3f}s")
                    print(f"   • Reallocation overhead: {realloc_pct:.1f}% of hierarchical step time")
                    
                    if realloc_pct > 80:
                        print(f"   ⚠️  Reallocation dominates hierarchical step - see reallocation benchmark above")
                
                stall_check_time = self.hierarchical_step_timings['stall_checking']
                stall_check_pct = (stall_check_time / hier_total_time) * 100
                if stall_check_pct > 10:
                    print(f"   • Stall checking: {stall_check_pct:.1f}% - relatively high overhead")
                
                # Overall time breakdown
                total_evolution_time = self.evolution_timings['total']
                total_hier_pct = (hier_total_time / total_evolution_time) * 100 if total_evolution_time > 0 else 0
                print(f"   • Hierarchical step overhead: {total_hier_pct:.1f}% of total evolution time")
                
            print("="*80 + "\n")


    def CreateEvolutionHistoryFile(self):
        """
        Create a file to store the evolution history of the swarm
        Insert header line
        """
        history_file_path = os.path.join(self.Output, "EnsembleEvolutionHistory.dat")

        # Check if directory exits, if not create it
        outdir = os.path.dirname(history_file_path)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Check if file already exists, if so overwrite.
        if os.path.isfile(history_file_path):
            print('Ensemble evolution file {} already exists, Overwriting'.format(history_file_path))
            os.system('rm {}'.format(history_file_path))

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
        import time
        t_start = time.time()
        self.hierarchical_step_call_count += 1

        # If all swarms are not stalled yet
        if self.AllStalled == False:
            
            t0 = time.time()
            swarms_to_freeze = []
            
            for swarm_index, Swarm_ in zip(list(self.Swarms.keys()),list(self.Swarms.values())):

                # If the mean of the spreads computed across the last 10 iterations has not gotten lower,
                #   Assume the swarm has stalled and thus conduct a hierarchical step

                if Swarm_.EvolutionCounter > self.Minimum_exploration_iterations and self.EvolutionCounter > self.Initial_exploration_limit:

                    if self.stall_condition(Swarm_):
                        swarms_to_freeze.append((swarm_index, Swarm_))
            
            self.hierarchical_step_timings['stall_checking'] += time.time() - t0
            
            # Freeze swarms that have stalled
            if swarms_to_freeze:
                t0 = time.time()
                for swarm_index, Swarm_ in swarms_to_freeze:
                    print('\n Swarm ',str(swarm_index),' reached stall condition, freezing')
                    
                    # Freeze until all swarms have been stalled in this given segment likelihood
                    self.frozen_swarms[swarm_index] = Swarm_
                    self.Swarms.pop(swarm_index)
                
                # If all the swarms have stalled:
                if len(list(self.Swarms.values())) == 0: 
                    self.AllStalled = True
                
                self.hierarchical_step_timings['swarm_freezing'] += time.time() - t0

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
                    
                    t0 = time.time()
                    self.Reallocate_particles()
                    self.hierarchical_step_timings['reallocation'] += time.time() - t0
                    self.reallocation_count += 1
        
        self.hierarchical_step_timings['total'] += time.time() - t_start


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
        
        # Print evolution benchmarks at the end
        if self.Verbose or self.evolution_call_count > 0:
            self.PrintEvolutionBenchmarks()

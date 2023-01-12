import os
import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import warnings
import copy
import dill as pickle

from .Model import Model
from PySO.Clustering_Swarms import Clustering
from .MWE_Swarm import Swarm as Swarm



class HierarchicalSwarmHandler(object):

    def __init__(self,
                 Hierarchical_models,
                 NumSwarms,
                 NumParticlesPerSwarm,
                 Omega = 0.6,
                 PhiP = 0.2,
                 PhiG = 0.2,
                 MH_fraction = 0.0,
                 Swarm_kwargs={},
                 Output = './',
                 nPeriodicCheckpoint = 10,
                 Swarm_names = None,
                 Verbose = False,
                 SaveEvolution = False,
                 Maxiter = 1e4,
                 Resume = True,
                 Maximum_number_of_iterations_per_step=400,
                 Minimum_exploration_iterations = 50,
                 Initial_exploration_limit= 150,
                 clustering_indices = None,
                 use_func_vals_in_clustering = False,
                 kick_velocities = True,
                 fitness_veto_fraction = 0.05,
                 max_particles_per_swarm = None,
                 redraw_velocities_at_segmentation = False):
        """

        REQUIRED INPUTS
        ------
        Hierarchical_models: list,
            list of Hierarchical model objects
        NumSwarms: int
            Number of Swarms to initialise.
        NumParticlesPerSwarm: list of ints,
            list containing number of particles to be assigned to each swarm. #


        OPTIONAL INPUTS
        ---------------
        Omega: float or list
            the omega parameter for each hierarhical model, inertial coefficient for velocity updating [defaults to .6]
        PhiP: float or list
            the phi_p parameter for each hierarhical model, cognitive coefficient for velocity updating [defaults to .2]
        PhiG: float or list
            the phi_g parameter for each hierarhical model, social coefficient for velocity updating [defaults to .2]
        MH_fraction: float:
            parameter controlling proportion of velocity rule dictated by MCMC, for each hierarchical model [defaults to 0.]
        Swarm_kwargs: dict,
            dictionary of common arguments between all swarms
        Output: str
            folder in which to save output [defaults to './']
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
            Minimum number of iterations to be done in each step before stall condition evaluated
        Initial_exploration_limit: int
            Minimum number of iterations done in the very first step before stall condition evaluated
        clustering_indices: None or list of int
            Parameter position indexes to use for relabelling/clustering step
        use_func_vals_in_clustering: boolean
            Boolean flag for using function values for clustering or not [defaults to False]
        kick_velocities: boolean
            Boolean flag for reinitialising velocities from position distribution
            on clustering and segmenting [defaults to True]
        fitness_veto_fraction: float
            Fraction of Best swarm position below which we throw away new swarms [defaults to 0.05]
        max_particles_per_swarm: integer
            Maximum number of particles per swarm [defaults to int(total_num_particles/10)]
        redraw_velocities_at_segmentation: Boolean
            Boolean flag indicating if the velocities should be redrawn from swarm position ranges at each segmentation step [defaults to False]
        """
        assert len(Hierarchical_models)>1, "Please input multiple models for Hierarchical PSO search"

        self.Hierarchical_models = Hierarchical_models

        self.Omegas = Omega
        self.PhiPs = PhiP
        self.PhiGs = PhiG
        self.MH_fractions = MH_fraction

        if type(self.Omegas) == float:
            self.Omegas = [self.Omegas] * len(self.Hierarchical_models)
            self.PhiPs = [self.PhiPs] * len(self.Hierarchical_models)
            self.PhiGs = [self.PhiGs] * len(self.Hierarchical_models)
            self.MH_fractions = [self.MH_fractions] * len(self.Hierarchical_models)
        else:
            assert len(self.Omegas) == len(self.Hierarchical_models), "Please ensure your PSO parameter lists correspond to the correct number of hierarchical steps "


        self.Model_axis_names = self.Hierarchical_models[1].names

        self.Ndim = len(self.Model_axis_names)

        self.NumSwarms = NumSwarms
        self.NumParticlesPerSwarm = NumParticlesPerSwarm
        self.Swarm_kwargs = Swarm_kwargs

        self.nPeriodicCheckpoint = nPeriodicCheckpoint

        self.BestKnownEnsemblePoint = np.zeros(self.Ndim)
        self.BestKnownEnsembleValue = None
        self.BestCurrentSwarm = None

        self.Verbose = Verbose

        self.SaveEvolution = SaveEvolution

        # Refering to the hierarchical steps
        self.Maximum_number_of_iterations_per_step = Maximum_number_of_iterations_per_step

        self.Minimum_exploration_iterations = Minimum_exploration_iterations

        self.Initial_exploration_limit = Initial_exploration_limit


        self.Maxiter = Maxiter

        self.Resume = Resume

        self.Output = Output

        self.Swarm_names = Swarm_names
        if self.Swarm_names == None: self.Swarm_names = np.arange(self.NumSwarms) # Defaults to numbered list of swarms


        self.clustering_indices  = clustering_indices
        if self.clustering_indices == None: self.clustering_indices = np.arange(self.Ndim) # Use all parameters by default in clustering

        self.use_func_vals_in_clustering = use_func_vals_in_clustering

        self.kick_velocities = kick_velocities

        self.redraw_velocities_at_segmentation = redraw_velocities_at_segmentation

        self.fitness_veto_fraction = fitness_veto_fraction

        self.max_particles_per_swarm = max_particles_per_swarm
        if self.max_particles_per_swarm == None: self.max_particles_per_swarm = int(self.NumParticlesPerSwarm/10)

        #Initialise swarms
        self.InitialiseSwarms()

        self.frozen_swarms = {}

        self.Hierarchical_model_counter = 0

        # Boolean variable to indicate if all swarms at the current level have stalled
        self.AllStalled = False

        self.swarm_stepping_done = False

        resume_file = os.path.join(self.Output, "PySO_resume.pkl")

        if self.Resume and os.path.exists(resume_file):
            print('Resuming from file {}'.format(resume_file))
            self.ResumeFromCheckpoint()


    def InitialiseSwarms(self):
        """
        Initialise the swarm points, values and velocities

        """
        self.Swarms = {self.Swarm_names[swarm_index]: Swarm(self.Hierarchical_models[0], self.NumParticlesPerSwarm,
                                                            Omega=self.Omegas[0], Phig= self.PhiGs[0], Phip=self.PhiPs[0], Mh_fraction=self.MH_fractions[0],
                                                            **self.Swarm_kwargs)
                       for swarm_index in self.Swarm_names}

        stability_num = self.stability_check(self.Omegas[0],
                                             self.PhiPs[0],
                                             self.PhiGs[0])
        print('Stability number:', stability_num)

        if stability_num <= 0:
            warnings.warn('Stability number is less than 0, initiated swarm is not guranteed to converge!')

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


    def EvolveSwarms(self):
        """
        Evolve every swarm through a single iteration
        """
        self.EvolutionCounter += 1

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
            # Check if the peak being explored is insignificant
            if (self.frozen_swarms[swarm_index].BestKnownSwarmValue - lowest_ensemble_val) < self.fitness_veto_fraction*(self.BestKnownEnsembleValue-lowest_ensemble_val):
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

                # if current size is over 25 it will just get bigger once reclustering occurs probably

                print('Swarm ',swarm_index, ' is over the maximum size per swarm, redistributing ',num_particles_in_swarm -
                      self.max_particles_per_swarm,' Particles')

        return(num_particles_redistributed)



    def Reallocate_particles(self):
        """Use all particles in current swarms, cluster them based on features and reallocate"""

        # Dont want to converge to or explore peaks that are below a certain threshold compared to the best of the entire ensemble
        # But dont want this redistribution to take place on the first "exploratory" swarm
        # Also dont want to redistribute if we are mostly doing MH MCMC velocity rule, as we dont expect strong clustering there
        if self.Hierarchical_model_counter != 0 and self.MH_fractions[self.Hierarchical_model_counter]< 0.1:
            num_particles_redistributed = self.veto_and_redistribute()

        # Extract positions to be used for clustering, only using the indices of parameters that are well measured AND function value     #
        # Feature_array - Particle positions, function values, not all components (Specially not ones well measured)
        # Chirp mass, distance, sky position, effective spin, time to merger, function values


        clustering_parameter_positions = np.array([np.take(self.frozen_swarms[swarm_index].Points,self.clustering_indices,axis=1) for swarm_index
                               in self.frozen_swarms.keys()])

        clustering_parameter_positions = np.concatenate(clustering_parameter_positions)


        if self.use_func_vals_in_clustering == True:
            clustering_function_values = np.array([self.frozen_swarms[swarm_index].Values for swarm_index
                                                   in self.frozen_swarms.keys()])
            clustering_function_values = np.concatenate(clustering_function_values)
            clustering_features = np.column_stack((clustering_parameter_positions, clustering_function_values))
        else:
            clustering_features = clustering_parameter_positions


        K, memberships = Clustering(clustering_features, min_membership=5,max_clusters=70)


        total_particle_positions = np.array([self.frozen_swarms[swarm_index].Points for swarm_index
                               in self.frozen_swarms.keys()])
        total_particle_velocities = np.array([self.frozen_swarms[swarm_index].Velocities for swarm_index
                               in self.frozen_swarms.keys()])

        total_particle_velocities = np.concatenate(total_particle_velocities)
        total_particle_positions = np.concatenate(total_particle_positions)

        print('Reinitiating swarms with Omega: ',self.Omegas[self.Hierarchical_model_counter+1],
              ' PhiP: ',self.PhiPs[self.Hierarchical_model_counter+1],
              ' PhiG: ',self.PhiGs[self.Hierarchical_model_counter+1])

        stability_num = self.stability_check(self.Omegas[self.Hierarchical_model_counter+1],
                                             self.PhiPs[self.Hierarchical_model_counter+1],
                                             self.PhiGs[self.Hierarchical_model_counter+1])
        print('Stability number:', stability_num)

        if stability_num<=0:
            warnings.warn('Stability number is less than 0, initiated swarm is not guranteed to converge!')


        for swarm_index in range(K):

              swarm_particle_positions = total_particle_positions[np.where(memberships == swarm_index)[0]]
              swarm_particle_velocities = total_particle_velocities[np.where(memberships == swarm_index)[0]]
              self.Swarms[swarm_index] = self.Reinitiate_swarm(swarm_particle_positions, swarm_particle_velocities)


        if self.Hierarchical_model_counter != 0 and num_particles_redistributed>0:
            # Redistribute particles from veto step above
            parameter_positions = np.array(
                [self.frozen_swarms[swarm_index].Points  for swarm_index
                 in self.frozen_swarms.keys()])

            parameter_positions = np.concatenate(parameter_positions)

            # Distribute them using covariance matrix of current clusters
            cov = np.cov(parameter_positions.T)
            position_mean = np.mean(parameter_positions,axis=0)
            velocity_mean = np.zeros(self.Ndim)

            redistributed_particle_positions = np.random.multivariate_normal(position_mean, cov, size=num_particles_redistributed)
            redistributed_particle_velocities = np.random.multivariate_normal(velocity_mean, cov, size=num_particles_redistributed)

            # Extra redistributed swarm
            self.Swarms[K] = self.Reinitiate_swarm(redistributed_particle_positions, redistributed_particle_velocities)
        self.frozen_swarms = {}
        self.AllStalled = False

        self.Hierarchical_model_counter += 1

    def stability_check(self, Omega, PhiP, PhiG):
        """
        Evaluate the stability number to diagnose if the swarm is diverging or converging.
        Definition of stability number and explaination provided in https://bee22.com/resources/Bergh%202006.pdf (Page 85, Equation 3.21)
                Omega > 1/2(PhiP + PhiG) − 1    (for guranteed convergence)

        INPUTS:
        ------
        Omega: float
            Inertia weight
        PhiP: float
            Personal/Cognitive weight
        PhiG: float
            Social weight

        RETURNS:
        ------
        stability_number: float
            Number defining the stability of the swarm being initiated
        """

        stability_number = Omega+1-1/2*(PhiP+PhiG)
        return(stability_number)

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

        num_particles = positions.shape[0]

        newswarm = Swarm(self.Hierarchical_models[self.Hierarchical_model_counter+ 1],num_particles,
                         Omega=Omega, Phip=PhiP, Phig=PhiG, Mh_fraction=MH_fraction , **self.Swarm_kwargs)

        newswarm.EvolutionCounter = 0
        newswarm.Points = positions

        if self.kick_velocities == True: #and self.Hierarchical_model_counter>0.5*len(self.Hierarchical_models):
            # Kick the reinitialised velocities
            # Regenerate velocities from a normal distribution specified by the covariance of particles in swarm
            vel_cov = np.cov(positions.T)

            # Reinitialising velocities with a mean of zero (Shape of mean is 1D with length number of dimensions)
            mean = np.zeros(positions.shape[1])

            # Draw velocities from normal distribution specified by position
            newswarm.Velocities = np.random.multivariate_normal(mean, vel_cov, size=num_particles)
        else:
            newswarm.Velocities = velocities

        if self.redraw_velocities_at_segmentation == True:

            ptp_vel_bounds = np.ptp(np.array([np.min(positions,axis=0),np.max(positions,axis=0)]).T,axis=1)

            newswarm.Velocities = (ptp_vel_bounds) * np.random.random_sample(size=(num_particles,self.Ndim)) - ptp_vel_bounds/2


        newswarm.BestKnownPoints = copy.deepcopy(newswarm.Points)

        # Recalculate best personal known values:
        for i in range(newswarm.NumParticles):
            newswarm.BestKnownValues[i] = newswarm.Model.log_likelihood(
                dict(zip(newswarm.Model.names, newswarm.Points[i])))

        newswarm.Values = copy.deepcopy(newswarm.BestKnownValues)

        newswarm.BestKnownSwarmPoint = newswarm.BestKnownPoints[np.argmax(newswarm.BestKnownValues)]
        newswarm.BestKnownSwarmValue = np.max(newswarm.BestKnownValues)

        # Sets up multiprocessing pool for parallel function computations
        newswarm.Pool = Pool(self.Swarm_kwargs['Nthreads'])

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

        # "# particle_number, name1, name2, name3, ..., function_value\n"
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

        # All the frozen swarm data
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
        final_swarm_values = np.concatenate(np.array([self.frozen_swarms[swarm_index].Values for swarm_index in list(self.frozen_swarms.keys())]))
        final_swarm_positions_filename = os.path.join(self.Output, "final_swarm_positions.txt")
        final_swarm_values_filename = os.path.join(self.Output, "final_swarm_values.txt")
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
        Checks if any of the swarms meet the condition to switch to the next model
        """


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
                    for swarm in self.Swarms:
                        swarm.Pool.close()
                        swarm.Pool.join()

                else:
                    print('\n All swarms stalled! Switching segments from ', str(self.Hierarchical_models[self.Hierarchical_model_counter].segment_number),
                          ' to ', str(self.Hierarchical_models[self.Hierarchical_model_counter+1].segment_number))
                    self.Reallocate_particles()


    def stall_condition(self,Swarm):
        """
        Evaluate stall condition for a given swarm
        """
        stalled = (((np.mean(Swarm.Spreads[-20:-10]) <= np.mean(Swarm.Spreads[-10:])) and
         # all(element == Swarm.FuncHistory[-20] for element in Swarm.FuncHistory[-20:]))
                   (np.abs(Swarm.FuncHistory[-1] - Swarm.FuncHistory[-20]) < 0.01*np.abs(Swarm.FuncHistory[-20]))) or
                    (Swarm.EvolutionCounter >=  self.Maximum_number_of_iterations_per_step ))
        return stalled


    def Run(self):
        """
        Run optmisation/sampling for all swarms
        """
        if self.SaveEvolution and self.EvolutionCounter==0:
            self.CreateEvolutionHistoryFile()
            self.SaveEnsembleEvolution()

        while self.ContinueCondition() and (self.swarm_stepping_done == False):
            self.EvolveSwarms()

            if self.EvolutionCounter % self.nPeriodicCheckpoint == 0:

                if self.Verbose: self.PrintStatus()

                if self.SaveEvolution: self.SaveEnsembleEvolution()
            self.check_hierarchical_step()

        self.SaveFinalResults()

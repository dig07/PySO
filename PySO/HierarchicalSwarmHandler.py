import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import copy

from .Model import Model
from PySO.Clustering_Swarms import Clustering
from .MWE_Swarm import MWE_Swarm as Swarm



class HierarchicalSwarmHandler(object):

    def __init__(self,
                 Hierarchical_models,
                 NumSwarms,
                 NumParticlesPerSwarm,
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
                 clustering_indices = None):
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
        """
        assert len(Hierarchical_models)>1, "Please input multiple models for Hierarchical PSO search"

        self.Hierarchical_models = Hierarchical_models

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
        #TODO need to allow the user to pass in the initial positions, velocities and target functions

        """
        self.Swarms = {self.Swarm_names[swarm_index]: Swarm(self.Hierarchical_models[0], self.NumParticlesPerSwarm, **self.Swarm_kwargs)
                       for swarm_index in range(len(self.Swarm_names))}
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

    def Reallocate_particles(self):
        """Use all particles in current swarms, cluster them based on features and reallocate"""

        # Extract positions to be used for clustering, only using the indices of parameters that are well measured AND function value     #
        # Feature_array - Particle positions, function values, not all components (Specially not ones well measured)
        # Chirp mass, distance, sky position, effective spin, time to merger, function values

        clustering_parameter_positions = np.array([np.take(self.frozen_swarms[swarm_index].Points,self.clustering_indices,axis=1) for swarm_index
                               in self.frozen_swarms.keys()])
        clustering_function_values = np.array([self.frozen_swarms[swarm_index].Values for swarm_index
                               in self.frozen_swarms.keys()])

        clustering_parameter_positions = np.concatenate(clustering_parameter_positions)
        clustering_function_values = np.concatenate(clustering_function_values)

        clustering_features = np.column_stack((clustering_parameter_positions, clustering_function_values))


        K, memberships = Clustering(clustering_features)


        total_particle_positions = np.array([self.frozen_swarms[swarm_index].Points for swarm_index
                               in self.frozen_swarms.keys()])
        total_particle_velocities = np.array([self.frozen_swarms[swarm_index].Velocities for swarm_index
                               in self.frozen_swarms.keys()])

        total_particle_velocities = np.concatenate(total_particle_velocities)
        total_particle_positions = np.concatenate(total_particle_positions)



        for swarm_index in range(K):

              swarm_particle_positions = total_particle_positions[np.where(memberships == swarm_index)[0]]
              swarm_particle_velocities = total_particle_velocities[np.where(memberships == swarm_index)[0]]
              if swarm_particle_velocities.shape[0] == 1: continue # Ignore clusters with just 1 particle in them
              self.Swarms[swarm_index] = self.Reinitiate_swarm(swarm_particle_positions, swarm_particle_velocities)

        self.frozen_swarms = {}
        self.AllStalled = False

        self.Hierarchical_model_counter += 1

    def Reinitiate_swarm(self,positions,velocities):
        """
        Reinitiate swarm given some positions and velocities.

        INPUTS:
        ------
        positions: array (number of particles, self.Ndim)
            initial positions for new swarm
        velocities: array (number of particles, self.Ndim)
            initial velocities for new swarm particles

        RETURNS:
        ------
        newswarm: swarm object
            new swarm initiated
        """

        num_particles = positions.shape[0]

        newswarm = Swarm(self.Hierarchical_models[self.Hierarchical_model_counter+ 1],
                         num_particles,
                         Verbose=False,
                         # Final two args mean evolution is saved at every iteration. Only necessary if running current_swarm.Plot()
                         SaveEvolution=False,  ############
                         Tol=self.Swarm_kwargs['Tol'], Nthreads=self.Swarm_kwargs['Nthreads'],
                         Omega=self.Swarm_kwargs['Omega'], PhiP=self.Swarm_kwargs['PhiP'],
                         PhiG=self.Swarm_kwargs['PhiG'], MaxIter=self.Maximum_number_of_iterations_per_step)

        newswarm.EvolutionCounter = 0
        newswarm.Points = positions
        newswarm.Velocities = velocities

        newswarm.BestKnownPoints = copy.deepcopy(newswarm.Points)

        # Recalculate best personal known values:
        for i in range(newswarm.NumParticles):
            newswarm.BestKnownValues[i] = newswarm.Model.log_likelihood(
                dict(zip(newswarm.Model.names, newswarm.Points[i])))

        newswarm.Values = copy.deepcopy(newswarm.BestKnownValues)

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
        header_string = header_string + "function_value\n"

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
                string += ",{}\n".format(self.Swarms[swarm_name].Values[particle_index])
                file = open(history_file_path, "a")
                file.write(string)
                file.close()

    def SaveFinalResults(self):
        """
        Save the final results to file
        """
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

                else:
                    print('\n All swarms stalled! Switching segments from ', str(self.Hierarchical_models[self.Hierarchical_model_counter].segment_number),
                          ' to ', str(self.Hierarchical_models[self.Hierarchical_model_counter+1].segment_number))
                    self.Reallocate_particles()


    def stall_condition(self,Swarm):
        """
        Evaluate stall condition for a given swarm
        """
        stalled = ((np.mean(Swarm.Spreads[-20:-10]) <= np.mean(Swarm.Spreads[-10:])) and
         all(element == Swarm.FuncHistory[-20] for element in Swarm.FuncHistory[-20:]))
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

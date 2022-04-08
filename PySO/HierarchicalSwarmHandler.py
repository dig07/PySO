import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import copy

from .Model import Model
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
                 Initial_exploration_limit= 150):
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
            Minimum number of iterations to be done in the exploring step before stall condition evaluated
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

        #Initialise swarms
        self.InitialiseSwarms()

        self.finished_swarms = {}

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

        # Feature_array - Particle positions, function values, not all components (Specially not ones well measured)

        # K, memberships = clustering(Features)

        # for swarm in range(K):

            # build new swarm for each K assign to dictionary

            # Summarise the clustering and put into save file somehow

            # Allow a few merges and allow quite a few more splits





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
        #Initialise each swarm and work out the initial best position for the ensemble
        for swarm_index, Swarm_ in enumerate(list(self.Swarms.values())):
            # If the mean of the spreads computed across the last 10 iterations has not gotten lower,
            #   Assume the system has stalled and thus conduct a hierarchical step

            if Swarm_.EvolutionCounter > self.Minimum_exploration_iterations and self.EvolutionCounter > self.Initial_exploration_limit:

                if self.stall_condition(Swarm_):

                    print('Swarm ',str(list(self.Swarms.keys())[swarm_index]),' reached stall condition, switching optimization functions')

                    if (Swarm_.Hierarchical_step+1) == len(self.Hierarchical_models):

                        # Save final results
                        Swarm_.SaveFinalResults()

                        self.finished_swarms[str(list(self.Swarms.keys())[swarm_index])]  = Swarm_

                        # Remove the swarm
                        self.Swarms.pop(list(self.Swarms.keys())[swarm_index])

                    else:
                        self.Swarms[swarm_index] = self.hierarchical_step(Swarm_)

        if len(list(self.Swarms.keys())) == 0:
            self.swarm_stepping_done = True


    def stall_condition(self,Swarm):
        """
        Evaluate stall condition for a given swarm
        """
        stalled = ((np.mean(Swarm.Spreads[-20:-10]) <= np.mean(Swarm.Spreads[-10:])) and
         all(element == Swarm.FuncHistory[-20] for element in Swarm.FuncHistory[-20:]))
        return stalled

    def hierarchical_step(self, current_swarm):
        """
        Generates a new swarm with the same positions and velocities but forgetting the previous history of old swarm.

        INPUTS:
        -------
        current_swarm: swarm object,
            current swarm to hierarchical step into new swarm

        RETURNS:
        -------
        newswarm: swarm object,
            new swarm object optimizing the new model

        """
        print('Switching from ',self.Hierarchical_models[current_swarm.Hierarchical_step].segment_number, ' to ',
              self.Hierarchical_models[current_swarm.Hierarchical_step+1].segment_number)
        newswarm = Swarm(self.Hierarchical_models[current_swarm.Hierarchical_step+1],
                                 current_swarm.NumParticles,
                                 Output=current_swarm.Output,
                                 Verbose=False,
                                 nPeriodicCheckpoint=1,
                                 # Final two args mean evolution is saved at every iteration. Only necessary if running current_swarm.Plot()
                                 SaveEvolution=False,  ############
                                 Tol=current_swarm.Tol, Nthreads=current_swarm.Nthreads,
                                 Omega=current_swarm.Omega, PhiP=current_swarm.PhiP,
                                 PhiG=current_swarm.PhiG, MaxIter=self.Maximum_number_of_iterations_per_step)
        newswarm.Hierarchical_step = current_swarm.Hierarchical_step + 1
        newswarm.EvolutionCounter = 0
        newswarm.Periodic = current_swarm.Periodic
        newswarm.Points = current_swarm.Points
        newswarm.Velocities = current_swarm.Velocities

        newswarm.BestKnownPoints = copy.deepcopy(current_swarm.Points)

        # Recalculate best personal known values:
        for i in range(newswarm.NumParticles):
            newswarm.BestKnownValues[i] = newswarm.Model.log_likelihood(
                dict(zip(newswarm.Model.names, newswarm.Points[i])))

        newswarm.Values = copy.deepcopy(newswarm.BestKnownValues)

        newswarm.BestKnownSwarmPoint = newswarm.BestKnownPoints[np.argmax(newswarm.BestKnownValues)]
        newswarm.BestKnownSwarmValue = np.max(newswarm.BestKnownValues)

        return (newswarm)

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

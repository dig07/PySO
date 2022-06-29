import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


from .Model import Model
from .MWE_Swarm import Swarm as Swarm




class SwarmHandler(object):

    def __init__(self,
                 Models,
                 NumSwarms,
                 NumParticlesPerSwarm,
                 Swarm_kwargs={},
                 Output = './',
                 nPeriodicCheckpoint = 10,
                 Swarm_names = None,
                 Verbose = False,
                 SaveEvolution = False,
                 Maxiter = 1e4,
                 Resume = True):
        """

        REQUIRED INPUTS
        ------

        Models: objects inheriting PySO.Model.Model, list or just an object
            this defines the problem)
            if just one object assume each swarm needs to target the same function to optimize/sample,
            if list we each swarm targets a different function
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
        """

        if type(Models) == list:
            self.Models = Models
        else:
            self.Models = [Models]*NumSwarms


        self.Model_axis_names = self.Models[1].names

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

        self.Maxiter = Maxiter

        self.Resume = Resume

        self.Output = Output

        self.Swarm_names = Swarm_names
        if self.Swarm_names == None: self.Swarm_names = np.arange(self.NumSwarms) # Defaults to numbered list of swarms


        #Initialise swarms
        self.InitialiseSwarms()

        resume_file = os.path.join(self.Output, "PySO_resume.pkl")

        if self.Resume and os.path.exists(resume_file):
            print('Resuming from file {}'.format(resume_file))
            self.ResumeFromCheckpoint()

    def InitialiseSwarms(self):
        """
        Initialise the swarm points, values and velocities
        #TODO need to allow the user to pass in the initial positions, velocities and target functions

        """
        self.Swarms = {self.Swarm_names[swarm_index]: Swarm(self.Models[swarm_index], self.NumParticlesPerSwarm, **self.Swarm_kwargs)
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

        for name in self.Swarm_names:
            self.Swarms[name].EvolveSwarm()

            if np.max(self.Swarms[name].BestKnownSwarmValue) > self.BestKnownEnsembleValue:
                self.BestKnownEnsembleValue = np.max(self.Swarms[name].BestKnownSwarmValue)
                self.BestKnownEnsemblePoint = self.Swarms[name].Points[np.argmax(self.Swarms[name].Values)]
                self.BestCurrentSwarm = name


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
        output_str += "Max Value: {0} ".format(self.BestKnownEnsembleValue)
        output_str += "at {0}, ".format(self.BestKnownEnsemblePoint)
        output_str += "from swarm {0}".format(self.BestCurrentSwarm)
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
        for swarm_name in self.Swarm_names:
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


    def Run(self):
        """
        Run optmisation/sampling for all swarms
        """
        if self.SaveEvolution and self.EvolutionCounter==0:
            self.CreateEvolutionHistoryFile()
            self.SaveEnsembleEvolution()

        while self.ContinueCondition():
            self.EvolveSwarms()

            if self.EvolutionCounter % self.nPeriodicCheckpoint == 0:

                if self.Verbose: self.PrintStatus()

                if self.SaveEvolution: self.SaveEnsembleEvolution()


        self.SaveFinalResults()

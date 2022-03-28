from abc import ABCMeta,abstractmethod,abstractproperty
from numpy import inf
from numpy.random import uniform

class Model(object):
    """
    Base class for user's model. User should subclass this
    and implement log_likelihood, names and bounds
    """
    __metaclass__ = ABCMeta
    names=[] # Names of parameters, e.g. ['p1','p2']
    bounds=[] # Bounds of prior as list of tuples, e.g. [(min1,max1), (min2,max2), ...]
    
    def in_bounds(self, param):
        """
        Checks whether param lies within the bounds
            
        -----------
        Parameters:
            param: dict
            
        -----------
        Return:
            True: if all dimensions are within the bounds
            False: otherwise
        """
        return all(self.bounds[i][0] < param[self.names[i]] < self.bounds[i][1] for i in range(len(self.names)))
    
    def new_point(self):
        """
        Create a new point, drawn from within bounds
            
        -----------
        Return:
            p: dict
        """
        p = dict({name: uniform(self.bounds[i][0],self.bounds[i][1])
                 for i, name in enumerate(self.names)})
        return p
    
    @abstractmethod
    def log_likelihood(self, param):
        """
        returns log likelihood of given parameter
            
        ------------
        Parameter:
            param: dict
        """
        pass
    
    def log_prior(self, param):
        """
        Returns log of prior.
            Default is flat prior within bounds
            
        ----------
        Parameter:
            param: :obj:`cpnest.parameter.LivePoint`
            
        ----------
        Return:
            0 if param is in bounds
            -np.inf otherwise
        """
        if self.in_bounds(param):
            return 0.0
        else: return -inf
    
    def log_posterior(self, param):
        """
        Returns log of posterior.
        
        ----------
        Parameter:
            param: dict
            
        ----------
        Return:
            log_prior + log_likelihood
        """
        return self.log_prior(param) + self.log_likelihood(param)

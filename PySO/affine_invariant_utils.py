import numpy as np

def ParallelStretchMove_InternalFunction(idx, my_half_swarm, other_half_swarm, Func, BoundsArray, velocities):

    Nparticles_other, Ndim = other_half_swarm.shape

    velocity = np.zeros(Ndim)

    # pick a particle from the other half of the swarm
    j = np.random.choice( range(Nparticles_other) )
    Xj = other_half_swarm[j] 

    # draw random variable z ~ g(z)
    z = sample_g()

    # Proposal point
    Y = Xj + z*(my_half_swarm[idx]-Xj)

    # acceptance probability
    logq = (Ndim-1)*np.log(z) + Func(Y) - Func(my_half_swarm[idx])

    # decide if we accept or reject
    r = np.random.uniform()
    in_bounds = ( np.all(BoundsArray[:,0]<Y) and np.all(Y<BoundsArray[:,1]) )
    accept = ( (np.log(r)<logq) and in_bounds )

    if accept:
        velocity = Y - my_half_swarm[idx]
    else: # reject
        pass

    return velocity


def invcdf_GW10(u, a=2.):
    """
    The inverse CDF of the distribution g(z) defined in 
    Eq.10 of D. Foreman-Mackey et al. 2013 (arXiv:1202.3665).

    INPUTS
    ------
    u: float or array
        uniformly distributed random numbers in range 0<u<1
    a: float
        the scale parameter a, defaults to 2

    RETURNS
    -------
    z: like u
        random samples z~g
    """
    z = (1.+(a-1.)*u)**2 / a
    return z

def sample_g(size=None, a=2.):
    """
    Return random samples from the distribution g(z) defined in 
    Eq.10 of D. Foreman-Mackey et al. 2013 (arXiv:1202.3665).

    A random variable 1/a < z < a from this distribution has a PDF
    $$ g(z) = \frac{\sqrt{a/z}}{2(a-1)}, $$
    and the PDF is zero for values outside of the allowed range.

    The mean and variance of this distribution are
    $$ E[z] = \frac{a^2+a+1}{3 a} $$
    and 
    $$ Var[z] = \frac{(a-1)^2 (a (4 a+7)+4)}{45 a^2} $$
    respectively.

    Because of the mild ~z^{-1/2} dependence of the PDF the PDF is 
    relatively flat across its entire range of support; the ratio of the
    maximum to the minimum value of the PDF across this range is a.

    Choosing a larger value of the scale parameter increases both the 
    mean and the range of support of the distribution, thereby making it
    more likely that a particle will propose a large step outside of the 
    current region of parameter space occupied by the swarm. 
    
    INPUTS
    ------
    size: int
        defaults to None, in which case a single sample is return
    a: float
        the scale parameter a, defaults to 2

    RETURNS
    -------
    samples: float or array
        if size is None, then a float is returned
        else if size is an int, then array of that length
    """
    u = np.random.uniform(size=size)
    samples = invcdf_GW10(u, a=a)
    return samples

import numpy as np
import scipy.linalg
from gillespie_utils import *

#########################################################################################################################################################################################
# steinberg-utils.py
# This library allows the user to calculate the Steinberg signature for a 3-vertex linear framework graph given any parameterization and test its sensitivity to increasing thermodynamic foce. The result of this investigation produces what we call force-area curves.
#########################################################################################################################################################################################

######################################################################################################################################################################################## 
# PARAMETER SAMPLING

# Determine initial parameters for the 3-vertex graph. This can be done logarithmically (sampling from 10^-3, 10^3) or "linearly" (sampling uniformly between a range created between a minimum and maximum value). Functions for paramter sets that satisfy detailed balance (equilibrium) and those that don't (non-equilibrium) included.

# Parameter order: w_12, w_21, w_23, w_32, w_13, w_31 = a, b, d, c, f, e
# Index order: omegas[0], omegas[1], omegas[2], omegas[3], omegas[4], omegas[5]
#########################################################################################################################################################################################

def log_eqparamsample_3vertex(min_val=-3,max_val=3,num_params=6):
    """
    Logarithmically samples equilibrium parameters for the 3-vertex graph from the range [10^min_val, 10^max_val].
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (10^min_val)
    max_val : scalar
        maximum value of sampling range (10^min_val)
    num_params: integer
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             parameter values in 3-state Markov process that satisfy the cycle condition
             order of parameters: w_12, w_21, w_23, w_32, w_13, w_31 = a, b, d, c, f, e = omegas[0], omegas[1], omegas[2], omegas[3], omegas[4], omegas[5]
    """
    omegas = np.zeros(num_params,dtype=np.float128)
    
    # choose the first 5 parameters at random
    omegas[:-1] = 10**(np.random.uniform(min_val,max_val, size = num_params-1))
    
    # allow the 6th parameter (omega_31) to be a free parameter
    # back-calculated with the cycle condition from the 3-vertex graph
    omegas[-1] = (omegas[1]*omegas[3]*omegas[4])/(omegas[0]*omegas[2])
                       
    return omegas

def log_noneqparamsample_3vertex(min_val=-3,max_val=3,num_params=6):
    """
    Logarithmically samples non-equilibrium parameters for the 3-vertex graph from the range [10^min_val, 10^max_val].
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (10^min_val)
    max_val : scalar
        maximum value of sampling range (10^max_val)
    num_params: integer
        number of rate constants in the Markov process (default=6)
               
    Returns
    -------
    omegas : 1D array
             non-equilibrium values of parameters in Markovian system
    """
    omegas = np.array([],dtype=np.float128)
    
    while omegas.size == 0:
                
        # choose 6 random parameters logarithmically
        vals = 10**(np.random.uniform(min_val,max_val, size = num_params))

        # calculate the forward and reverse cycle products
        forward = vals[0]*vals[2]*vals[5]
        reverse = vals[1]*vals[3]*vals[4]
        
        # if they don't satisfy detailed balance (fat chance), let them be the omegas
        if (forward != reverse) and (reverse != 0):
            omegas = vals
    
    return omegas

def lin_eqparamsample_3vertex(min_val=0.001,max_val=100,num_params=6):
    """
    Randomly samples equilibrium parameter sets in the range [0.001,100] for a 3-state Markov process.
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (default=0.001)
    max_val : scalar
        maximum value of sampling range (default=100)
    num_params: integer
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             equilibrium values of parameters in Markovian system
    """
    omegas = np.zeros(num_params)
    
    # choose the first 5 parameters at random
    omegas[:-1] = np.around(np.random.choice(np.arange(min_val,max_val,step = 0.001),size=num_params-1),3)
    
    # allow the 6th parameter (omega_31) to be a free parameter
    # back-calculated with the cycle condition from the 3-vertex graph
    omegas[-1] = (omegas[1]*omegas[3]*omegas[4])/(omegas[0]*omegas[2])
                       
    return omegas

def lin_noneqparamsample_3vertex(min_val=0.001, max_val=100,num_params=6):
    """
    Randomly samples non-equilibrium parameter sets in the range [0.001,100] for a 3-state Markov process.
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (default=0.001)
    max_val : scalar
        maximum value of sampling range (default=100)
    num_params: integer
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             non-equilibrium values of parameters in Markovian system
    """
    omegas = np.array([],dtype=np.float128)
    
    while omegas.size == 0:
        
        # choose 6 random integers betweem 0 and 200
        vals = np.around(np.random.choice(np.arange(min_val,max_val,step = 0.001),size=num_params),3)

        # calculate the forward and reverse cycle products
        forward = vals[0]*vals[2]*vals[5]
        reverse = vals[1]*vals[3]*vals[4]
        
        # if they don't satisfy detailed balance (fat chance), let them be the omegas        
        if (forward != reverse) and (reverse != 0):
            omegas = vals
    
    return omegas

#########################################################################################################################################################################################
# SELECT A PARAMETER TO ALTER

# To test the sensitivity of the Steinberg signature to 
#########################################################################################################################################################################################

def param_choice(num_params=6):
    """
    Randomly determines the index of the parameter that will be perturbed from its equilibrium value.
    
    Parameters
    ----------
    num_params: integer
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    idx : integer
        index of parameter in omegas that will be perturbed from its equilibrium value
        order of parameters: w_12, w_21, w_23, w_32, w_13, w_31 = a, b, d, c, f, e = omegas[0], omegas[1], omegas[2], omegas[3], omegas[4], omegas[5]
    """
    
    idx = np.random.choice(np.arange(num_params-1))
    
    return np.random.choice(np.arange(num_params-1))

#########################################################################################################################################################################################
# Functions to calculate the Laplacian matrix from a given set of parameters
#########################################################################################################################################################################################

def Laplacian_3state(omegas):
    """
    Randomly samples equilibrium parameter sets in the range [0.01,100] for a 4-state Markov process.
    
    Parameters
    ----------
    omegas : 1D array
            parameter values of rate constants in 4-state Markovian system
            omegas = [a,b,d,c,f,e] = [0,1,2,3,4,5]
        
    Returns
    -------
    L : 2D array
        column-based Laplacian matrix of 4-state Markovian system
    """
    
    L = np.array([[-(omegas[0]+omegas[4]), omegas[1], omegas[5]], [omegas[0], -(omegas[1]+omegas[2]), omegas[3]], [omegas[4], omegas[2], -(omegas[5]+omegas[3])]],dtype=np.float128)
    
    return L

def Laplacian_4state(omegas):
    """
    Randomly samples equilibrium parameter sets in the range [0.01,100] for a 4-state Markov process.
    
    Parameters
    ----------
    omegas : 1D array
            parameter values of rate constants in 4-state Markovian system
            omegas = [k_12,k_21,k_14,k_41,k_42,k_24,k_32,k_23,k_34,k_43]
        
    Returns
    -------
    L : 2D array
        column-based Laplacian matrix of 4-state Markovian system
    """
    
    L = np.array([[-omegas[0]-omegas[2], omegas[1], 0, omegas[3]],
                  [omegas[0], -omegas[1]-omegas[7]-omegas[5], omegas[6], omegas[4]],
                  [0, omegas[7], -omegas[6]-omegas[8], omegas[9]],
                  [omegas[2], omegas[5], omegas[8], -omegas[3]-omegas[4]-omegas[9]]],dtype=np.float128)
    
    return L

#########################################################################################################################################################################################
# Functions calculating the affinity from a set of parameters.
#########################################################################################################################################################################################

def cycle_affinity_3state(omegas):
    """
    Calculates the cycle affinity (or the thermodynamic force) for a single cycle, 3 state Markov process
    
    Parameters
    ----------
    omegas : 1D array
             parameter values of the system
    
    Returns
    -------
    affinity : scalar
               value of the thermodynamic foce of the system
    """
    
    # calculate the forward and reverse cycle products
    forward = omegas[0]*omegas[2]*omegas[5]
    reverse = omegas[1]*omegas[3]*omegas[4]
    
    # calculate the cycle affinity
    affinity = np.log(forward/reverse)
    
    return affinity

#########################################################################################################################################################################################
# Functions calculating the higher order autocorrelation functions
#########################################################################################################################################################################################

def NG_III_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
    """
    Calculates the analytical solution for autocorrelation function given a Laplacian matrix
    
    Parameters
    ----------
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    L : 2D array
        column-based Laplacian matrix of system (including diagonal entries)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    alpha : scalar
        exponent
    beta : scalar
        exponent
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    
    """
    f = np.array([observable],dtype=np.float128)
    fstar = f.T
    
    eigvals, eigvecs = scipy.linalg.eig(L)
    pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)],dtype=np.float128).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n),dtype=np.float128)
    t_rev = np.zeros(len(tau_n),dtype=np.float128)
    
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ list_result[i] @(fstar ** beta * pi)
        t_rev[i] = f**beta @ list_result[i] @(fstar ** alpha * pi)
        
    return t, t_rev

#########################################################################################################################################################################################
# Functions calculating the Steinberg signature
#########################################################################################################################################################################################

def steinberg_signature(t,t_rev):
    return np.abs(np.trapz(t)-np.trapz(t_rev))

#########################################################################################################################################################################################
# Force area analysis
#########################################################################################################################################################################################

def peturbation(omegas, param_choice, m=1.01):
    
    omegas[param_choice] = omegas[param_choice]*m
    
    return omegas

def force_area(num_perturbations, omegas, param_choice, observable, tau_n,m=1.01):
    
    forces = np.zeros(num_perturbations,dtype=np.float128)
    areas = np.zeros(num_perturbations,dtype=np.float128)
    
    for i in range(num_perturbations):

        # calculate the cycle affinity        
        forces[i] = cycle_affinity_3state(omegas)
        
        L = np.array([[-(omegas[0]+omegas[4]), omegas[1], omegas[5]], [omegas[0], -(omegas[1]+omegas[2]), omegas[3]], [omegas[4], omegas[2], -(omegas[5]+omegas[3])]],dtype=np.float128)
        
        t, t_rev = NG_III_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)
        
        areas[i] = np.abs(np.trapz(t)-np.trapz(t_rev))
        
        # modify the value of one parameter
        omegas = omegas[param_choice]*m
    
    return forces, areas

### from other files

def cycle_affinity_3state(omegas):
    """
    Calculates the cycle affinity (or the thermodynamic force) for a single cycle, 3 state Markov process
    
    Parameters
    ----------
    omegas : 1D array
             parameter values of the system
    
    Returns
    -------
    affinity : scalar
               value of the thermodynamic foce of the system
    """
    
    # calculate the forward and reverse cycle products
    forward = omegas[0]*omegas[2]*omegas[5]
    reverse = omegas[1]*omegas[3]*omegas[4]
    
    # calculate the cycle affinity
    affinity = np.log(forward/reverse)
    
    return affinity

def NG_III_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
    """
    Calculates the analytical solution for autocorrelation function given a Laplacian matrix
    
    Parameters
    ----------
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    L : 2D array
        column-based Laplacian matrix of system (including diagonal entries)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    alpha : scalar
        exponent
    beta : scalar
        exponent
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    
    """
    f = np.array([observable])
    fstar = f.T
    
    eigvals, eigvecs = np.linalg.eig(L)
    pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)]).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n))
    t_rev = np.zeros(len(tau_n))
    
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ list_result[i] @(fstar ** beta * pi)
        t_rev[i] = f**beta @ list_result[i] @(fstar ** alpha * pi)
        
    return t, t_rev

"""
Autocorrelation function calculations
"""

def NG_III_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
    """
    Calculates the analytical solution for autocorrelation function given a Laplacian matrix
    
    Parameters
    ----------
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    L : 2D array
        column-based Laplacian matrix of system (including diagonal entries)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    alpha : scalar
        exponent
    beta : scalar
        exponent
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    
    """
    f = np.array([observable])
    fstar = f.T
    
    eigvals, eigvecs = np.linalg.eig(L)
    pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)]).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n))
    t_rev = np.zeros(len(tau_n))
    
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ list_result[i] @(fstar ** beta * pi)
        t_rev[i] = f**beta @ list_result[i] @(fstar ** alpha * pi)
        
    return t, t_rev

def autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
    """
    Calculates the analytical solution for forward and reverse higher-order autocorrelation functions for a particular Laplacian matrix
    
    Parameters
    ----------
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
        
    L : NxN array
        column-based Laplacian matrix of linear framework graph with N vertices
    
    tau_n : 1D array
        range of intervals between values of observable taken by system
    
    alpha : scalar
        exponent applied to observable
    
    beta : scalar
        exponent applied to transpose of observable
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    
    t_rev : 1D array
        reverse autocorrelation function values
    
    """
    f = np.array([observable],dtype=np.float128)
    fstar = f.T
    
    # calculate the stationary distribution of the Markov process
    eigvals, eigvecs = scipy.linalg.eig(L)
    pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)]).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n),dtype=np.float128)
    t_rev = np.zeros(len(tau_n),dtype=np.float128)
    
    # multiply a copy of the Laplacian against each value of tau and put it into a list
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ list_result[i] @(fstar ** beta * pi)
        t_rev[i] = f**beta @ list_result[i] @(fstar ** alpha * pi)
        
    return t, t_rev


def old_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
    """
    Calculates the analytical solution for autocorrelation function given a Laplacian matrix
    
    Parameters
    ----------
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    L : 2D array
        column-based Laplacian matrix of system (including diagonal entries)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    alpha : scalar
        exponent
    beta : scalar
        exponent
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    
    """
    f = np.array([observable])
    fstar = f.T
    
    eigvals = scipy.linalg.eig(L,left=True,right=True)[0]
    right_eigvecs = scipy.linalg.eig(L,left=True,right=True)[2]
    pi = np.array([right_eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(right_eigvecs[:,np.argmin(np.abs(eigvals))].real)]).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n))
    t_rev = np.zeros(len(tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ scipy.linalg.expm(L*tau_n[i]) @(fstar ** beta * pi)
        t_rev[i] = f**beta @ scipy.linalg.expm(L*tau_n[i]) @(fstar ** alpha * pi)
        
    return t, t_rev

def autocorrelation_eq_compare_log(observable,tau_n,min_val=-3,max_val=3):
    """
    Randomly samples parameters that satisfy the cycle condition, and calculates the values of G and G_r,
    the cycle affinity, and the area between G and G_r
    
    Parameters
    ----------
    max_val : scalar
        maximum value of parameter sampling range
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    
    Returns
    -------
    omegas : 1D array
        equilibrium values of parameters in Markovian system
    affinity : scalar
        value of the thermodynamic foce of the system
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    area : scalar
        area between the forward (G) and reverse (G_r) autocorrelation functions
    """
    
    # sample equilibrium parameters
    omegas = log_eqparamsample_3vertex(min_val,max_val)
    
    affinity = cycle_affinity_3state(omegas)
    
    # calculate the Laplacian from params
    R = np.array([[0, omegas[0], omegas[4]], [omegas[1], 0, omegas[2]], [omegas[5], omegas[3], 0]])
    L = np.transpose(R) + np.array([[-(omegas[0]+omegas[4]), 0, 0], [0, -(omegas[1]+omegas[2]), 0], [0, 0, -(omegas[5]+omegas[3])]])
    
    # calculate the autocorrelation function values in forward and reverse
    t, t_rev = NG_III_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)
    
    # calculate the area between them
    area = np.abs(np.trapz(t)-np.trapz(t_rev))
    
    return omegas, affinity, t, t_rev, area

def autocorrelation_eq_compare_lin(max_val,observable,tau_n):
    """
    Randomly samples parameters that satisfy the cycle condition, and calculates the values of G and G_r,
    the cycle affinity, and the area between G and G_r
    
    Parameters
    ----------
    max_val : scalar
        maximum value of parameter sampling range
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    
    Returns
    -------
    omegas : 1D array
        equilibrium values of parameters in Markovian system
    affinity : scalar
        value of the thermodynamic foce of the system
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    area : scalar
        area between the forward (G) and reverse (G_r) autocorrelation functions
    """
    
    # sample equilibrium parameters
    omegas = lin_eqparamsample_3vertex(max_val)
    
    affinity = cycle_affinity_3state(omegas)
    
    # calculate the Laplacian from params
    R = np.array([[0, omegas[0], omegas[4]], [omegas[1], 0, omegas[2]], [omegas[5], omegas[3], 0]])
    L = np.transpose(R) + np.array([[-(omegas[0]+omegas[4]), 0, 0], [0, -(omegas[1]+omegas[2]), 0], [0, 0, -(omegas[5]+omegas[3])]])
    
    # calculate the autocorrelation function values in forward and reverse
    t, t_rev = NG_III_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)
    
    # calculate the area between them
    area = np.abs(np.trapz(t)-np.trapz(t_rev))
    
    return omegas, affinity, t, t_rev, area

def autocorrelation_noneq_compare(observable,tau_n,min_val=-3,max_val=3):
    """
    Randomly samples parameters that satisfy the cycle condition, and calculates the values of G and G_r,
    the cycle affinity, and the area between G and G_r
    
    Parameters
    ----------
    max_val : scalar
        maximum value of parameter sampling range
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    
    Returns
    -------
    omegas : 1D array
        equilibrium values of parameters in Markovian system
    affinity : scalar
        value of the thermodynamic foce of the system
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    area : scalar
        area between the forward (G) and reverse (G_r) autocorrelation functions
    """
    
    # sample equilibrium parameters
    omegas = log_noneqparamsample_3vertex(min_val=-3,max_val=3)
    
    affinity = cycle_affinity_3state(omegas)
    
    # calculate the Laplacian from params
    R = np.array([[0, omegas[0], omegas[4]], [omegas[1], 0, omegas[2]], [omegas[5], omegas[3], 0]])
    L = np.transpose(R) + np.array([[-(omegas[0]+omegas[4]), 0, 0], [0, -(omegas[1]+omegas[2]), 0], [0, 0, -(omegas[5]+omegas[3])]])
    
    # calculate the autocorrelation function values in forward and reverse
    t, t_rev = autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)
    
    # calculate the area between them
    area = np.abs(np.trapz(t)-np.trapz(t_rev))
    
    return omegas, affinity, t, t_rev, area

def area_force_compare(max_val,observable,tau_n):
    """
    Randomly samples parameters that satisfy the cycle condition, and compares the cycle affinity and
    the area between forward/reverse autocorrelation functions as one paramter is perturbed from its 
    equilibrium value
    
    Parameters
    ----------
    max_val : scalar
        maximum value of parameter sampling range
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    
    Returns
    -------
    affinities : 1D array
        array containing cycle affinities of system as it is perturbed from equilibrium 
    areas : 1D array
        area between the forward and reverse autocorrelation functions as system is perturbed from equilibrium
    """
    
    # define arrays
    affinities = np.array([])
    areas = np.array([])

    for i in range(0,1000):
        # sample equilibrium parameters
        omegas = eq_param_sample(max_val)
        
        # choose one parameter to vary and perturb it from equilibrium value
        param_choice = np.random.choice(np.arange(0,5),size=1)[0]
        omegas[param_choice] += 1
        
        # calculate the cycle affinity and add to affinities
        affinity = cycle_affinity_3state(omegas)
        affinities = np.append(affinities, affinity)
        
        # calculate the Laplacian from params
        R = np.array([[0, omegas[0], omegas[4]], [omegas[1], 0, omegas[2]], [omegas[5], omegas[3], 0]])
        L = np.transpose(R) + np.array([[-(omegas[0]+omegas[4]), 0, 0], [0, -(omegas[1]+omegas[2]), 0], [0, 0, -(omegas[5]+omegas[3])]])

        # calculate forward and reverse autocorrelation function values
        t, t_rev = autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)

        # calculate the area between them and add to areas
        area = np.abs(np.trapz(t)-np.trapz(t_rev))
        areas = np.append(areas, area)
        
        return affinities, areas

def nearest_idx(array,value,axis=None):
    """
    Return the index of the array that is closest to value. Equivalent to
    `np.argmin(np.abs(array-value))`
    
    Parameters
    ----------
    array : arraylike
    value : Scalar
    axis  : int, optional
        From np.argmin: "By default, the index is into the flattened array, otherwise
        along the specified axis."
    Returns
    -------
    idx : IndexArray
        index of array that most closely matches value. If axis=None this will be an integer
    """
    array = np.asarray(array)
    return np.argmin(np.abs(array-value),axis=axis)

def gillespie_square_wave(x,f,max_time,observable,tau,target_probs,num_states,min_time_step=0.01):
    """
    calculate the gillespie trajectory filling in constant values for the times
    between transitions.
    
    Parameters
    ----------
    x : scalar
        initial state
    f: scalar
        initial value of obvservable
    max_time: scalar
        maximum time to simulate untill
    min_time_step: scalar
        
    
    Returns
    -------
    times : 1D array
        time spent in each state
    traj_obs : 1D array
        sequence of observable values
    traj_states: 1D array
        sequence of states
    """
    # trajectory arrays
    N_time_points = np.int(max_time/min_time_step) # number of time points
    traj_states = np.zeros(N_time_points) # sequence of states
    # traj_time = np.zeros(N_time_points) # time spent in each state
    traj_obs = np.zeros(N_time_points) # values of the observable
    times = np.linspace(0,max_time,N_time_points)
    previous_time = 0
    cur_time = 0
    t_idx = 0
    
    while cur_time < max_time:
        previous_time = cur_time
        cur_time += draw_time(x,tau)
        start_idx = nearest_idx(times,previous_time)
        end_idx = np.clip(nearest_idx(times,cur_time),0,N_time_points)
        traj_states[start_idx:end_idx] = x
        traj_obs[start_idx:end_idx] = observable[x]
        x = draw_target(x,target_probs,num_states)

    return times, traj_obs, traj_states

def correlate_alpha_beta(x,alpha,beta):
    """
    calculate the correlation of $f(t)**alpha * f(t+τ)**beta
    parameters
    ----------
    x : 1D array
    alpha : scalar
    beta : scalar
    
    """
    result = np.correlate(x**alpha, x**beta, mode='full')
    return result[np.int(result.size/2):]

"""
Functions for sampling parameters for the 3-vertex graph in equilibrium or non-equilibrium steady states. Functions are included for logarithmic or unifrom sampling of parameter values.
"""

def log_eqparamsample_3vertex(min_val=-3,max_val=3,num_params=6):
    """
    Logarithmically samples equilibrium parameters for the 3-vertex graph from the range [10^min_val, 10^max_val].
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (10^min_val)
    max_val : scalar
        maximum value of sampling range (10^min_val)
    num_params: scalar
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             parameter values in 3-state Markov process that satisfy the cycle condition
             order of parameters: 
    """
    omegas = np.zeros(num_params)
    
    # choose the first 5 parameters at random
    omegas[:-1] = 10**(np.random.uniform(min_val,max_val, size = num_params-1))
    
    # allow the 6th parameter (omega_31) to be a free parameter
    omegas[-1] = (omegas[1]*omegas[3]*omegas[4])/(omegas[0]*omegas[2])
                       
    return omegas

def log_noneqparamsample_3vertex(min_val=-3,max_val=3,num_params=6):
    """
    Logarithmically samples non-equilibrium parameters for the 3-vertex graph from the range [10^min_val, 10^max_val].
    
    Parameters
    ----------
    min_val : scalar›
        minimum value of sampling range (10^min_val)
    max_val : scalar
        maximum value of sampling range (10^max_val)
    num_params: scalar
        number of states in the Markov process (default=6)
               
    Returns
    -------
    omegas : 1D array
             non-equilibrium values of parameters in Markovian system
    """
    omegas = np.array([])
    
    while omegas.size == 0:
        
        # choose 6 random integers betweem 0 and 200
        
        # vals.fill(np.random.choice(np.arange(min_val, max_val,step=min_val)))
        vals = 10**(np.random.uniform(min_val,max_val, size = num_params))

        # calculate the forward and reverse cycle products
        forward = vals[0]*vals[2]*vals[5]
        reverse = vals[1]*vals[3]*vals[4]
        
        if (forward != reverse) and (reverse != 0):
            omegas = vals
    
    return omegas

def lin_eqparamsample_3vertex(min_val=0.001,max_val=100,num_params=6):
    """
    Randomly samples equilibrium parameter sets in the range [0.01,100] for a 3-state Markov process.
    
    Parameters
    ----------
    max_val : scalar
        maximum value of sampling range (default=100)
    num_params: scalar
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             equilibrium values of parameters in Markovian system
    """
    omegas = np.zeros(num_params)
    
    # choose the first 5 parameters at random
    omegas[:-1] = np.around(np.random.choice(np.arange(min_val,max_val,step = 0.001),size=num_params-1),3)
    
    # allow the 6th parameter (omega_31) to be a free parameter
    omegas[-1] = (omegas[1]*omegas[3]*omegas[4])/(omegas[0]*omegas[2])
                       
    return omegas

def lin_noneqparamsample_3vertex(min_val=0.001, max_val=100,num_params=6):
    """
    Randomly samples non-equilibrium parameter sets in the range [0.001,100] for a 3-state Markov process.
    
    Parameters
    ----------
    max_val : scalar
        maximum value of sampling range (default=100)
    num_params: scalar
        number of states in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             non-equilibrium values of parameters in Markovian system
    """
    omegas = np.array([])
    
    while omegas.size == 0:
        
        # choose 6 random integers betweem 0 and 200
        vals = np.around(np.random.choice(np.arange(min_val,max_val,step = 0.001),size=num_params),3)

        # calculate the forward and reverse cycle products
        forward = vals[0]*vals[2]*vals[5]
        reverse = vals[1]*vals[3]*vals[4]
        
        if (forward != reverse) and (reverse != 0):
            omegas = vals
    
    return omegas

def is_noisy(array):
    """
    Determines if the values in an array are noisy.
    
    Args:
        array: The array to be checked. 
    Returns:
        True if the values in the array are noisy, False otherwise.
    """
    # Calculate the standard deviation of the array.
    stddev = np.std(array)
    # Calculate the mean of the array.
    mean = np.mean(array)
    # Check if the standard deviation is greater than a certain threshold.
    # This threshold can be adjusted to control how sensitive the function is to noise.
    if stddev > 0.1:
        return True
    else:
        return False
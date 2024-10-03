import numpy as np
import scipy.linalg

# steinberg_utils_3vertex.py

# This library allows the user to compute higher-order autocorrelation functions and the Steinberg signature for any linear framework graph (which represent continuous-time, finite-state, time-homogeneous Markov process). For relevant mathematical details, please refer to the accompanying manuscript Haque, Cetiner, Gunawardena 2024. The code in this file is designed with the 3-vertex graph, K, as our analysis was primarily performed for this system. That being said, some of the functions in this library are able to be used on linear framework graphs of any size. For code optimized for handling general graphs, please see the library `general_graphs.py`. 

## PARAMETER SAMPLING ##

# Randomly sampling transition rates for the 3-vertex graph, K. These parameters are defined as 10^x, where x is randomly drawn from the uniform distribution on (-3, 3). Here, we provide a function for randomly sampling parameters that satisfy detailed balance and one for which the parameters do not necessarily satisfy detailed balance.

# See Figure 1A. in Haque, Cetiner, Gunawardena 2024 for symbolic edge label assignments. The array params lists these edge labels in the following order: a, b, d, c, f, e.

def equilibrium_parameters(min_val=-3,max_val=3,num_params=6):
    """
    Randomly samples transition rates for a 3-vertex graph, K, which satisfy detailed balance. These parameters are defined as 10^x, where x is randomly drawn from the uniform distribution on (min_val, max_val). 
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (default=-3)
    max_val : scalar
        maximum value of sampling range (default=3)
    num_params: integer
        number of transition rates in graph (default=6)
        
    Returns
    -------
    params : 1D array
             transition rates in the 3-vertex graph K satisfy the cycle condition
             order of parameters: a, b, d, c, f, e = params[0], params[1], params[2], params[3], params[4], params[5]
    """
    params = np.zeros(num_params,dtype=np.float128)
    
    # randomly sample the first 5 parameters
    params[:-1] = 10**(np.random.uniform(min_val,max_val, size = num_params-1))
    
    # allow the 6th parameter (e = params[-1]) to be a free parameter
    params[-1] = (params[1]*params[3]*params[4])/(params[0]*params[2])
                       
    return params

def random_parameters(min_val=-3,max_val=3,num_params=6):
    """
    Randomly samples transition rates for a 3-vertex graph, K, which do not necessarily satisfy detailed balance. These parameters are defined as 10^x, where x is randomly drawn from the uniform distribution on (min_val, max_val).
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (default=-3)
    max_val : scalar
        maximum value of sampling range (default=3)
    num_params: integer
        number of transition rates in graph (default=6)
               
    Returns
    -------
    params : 1D array
             transition rates in the 3-vertex graph K
             order of parameters: a, b, d, c, f, e = params[0], params[1], params[2], params[3], params[4], params[5]
    """
    
    params = np.zeros(num_params,dtype=np.float128)
    
    params[:] = 10**(np.random.uniform(min_val,max_val, size = num_params))
    
    return params

## LAPLACIAN MATRIX & STEADY STATE DISTRIBUTION ##

# See Figure 1B. in Haque, Cetiner, Gunawardena 2024 for the Laplacian matrix of the 3-vertex graph K.

def Laplacian_K(params):
    """
    Calculates the Laplacian matrix for the 3-vertex graph K.
    
    Parameters
    ----------
    params : 1D array
             transition rates in the 3-vertex graph K
             order of parameters: a, b, d, c, f, e = params[0], params[1], params[2], params[3], params[4], params[5]
    
    Returns
    -------
    L : 3x3 array
        column-based Laplacian matrix of 3-vertex graph K
    """
    a = params[0]
    b = params[1]
    d = params[2]
    c = params[3]
    f = params[4]
    e = params[5]
    
    L = np.array([[-(a+f), b, e], [a, -(b+d), c], [f, d, -(e+c)]],dtype=np.float128)
    
    return L

def steady_state_MTT_K(params):
    """
    Calculates the steady-state distribution for the 3-vertex graph K using the linear framework and the Matrix-Tree Theorem.
    
    Parameters
    ----------
    params : 1D array
             parameter values of rate constants in 3-vertex graph K
             params = [a,b,d,c,f,e]
    
    Returns
    -------
    pi : 1D array
         the steady state distribution for a 3-vertex graph K.
    
    """
    a = params[0]
    b = params[1]
    d = params[2]
    c = params[3]
    f = params[4]
    e = params[5]
    
    rho_1 = b*c + b*e + d*e
    
    rho_2 = a*e + a*c + c*f
    
    rho_3 = a*d + b*f + d*f
    
    rho_tot = rho_1 + rho_2 + rho_3
    
    pi = np.array([rho_1/rho_tot, rho_2/rho_tot, rho_3/rho_tot])
    
    return pi

## CYCLE AFFINITY ##

# See equation 15 in Haque, Cetiner, Gunawardena 2024. For the 3-vertex graph K, A(C) = ln(ade/bfc).

def cycle_affinity_K(params):
    """
    Calculates the cycle affinity for a 3-vertex graph K.
    
    Parameters
    ----------
    params : 1D array
             parameter values of rate constants in 3-vertex graph K
             params = [a,b,d,c,f,e]
             
    Returns
    -------
    affinity : scalar
               cycle affinity
    """
    
    a = params[0]
    b = params[1]
    d = params[2]
    c = params[3]
    f = params[4]
    e = params[5]
    
    # calculate the forward and reverse cycle products
    forward = a*d*e
    reverse = b*c*f
    
    # calculate the cycle affinity
    affinity = np.abs(np.log(forward/reverse))
    
    return affinity

## HIGHER-ORDER AUTOCORRELATION FUNCTIONS ##

def asymmetric_autocorrelation(signal,L,tau,pi,alpha=1,beta=3):
    """
    Numerically calculates the asymmetric autocorrelation functions A^{1,3}(\tau) and A^{3,1}(\tau) for a particular Laplacian matrix. This function works for a linear framework graph of any size.
    
    Parameters
    ----------
    signal : 1D array
        vector of possible values of signal S = (S(1), ..., S(N))
        
    L : NxN array
        column-based Laplacian matrix of linear framework graph with N vertices
    
    tau : 1D array
        range of intervals between values of signal along integration interval
        
    pi : 1D array
         the steady state distribution for a linear framework graph with N vertices
    
    alpha, beta : scalar
        asymmetric exponents applied to signal (default: alpha=1, beta=3)
    
    Returns
    -------
    a_13 : 1D array
        forward autocorrelation function values
    
    a_31 : 1D array
        reverse autocorrelation function values
    
    """
    # initialize forward and reverse autocorrelation function arrays
    a_13 = np.zeros(len(tau),dtype=np.float128)
    a_31 = np.zeros(len(tau),dtype=np.float128)
    
    # define the signal vectors
    # define the signal vectors
    s_t = np.array([signal],dtype=np.float128) # row vector
    s = s_t.T # column vector
    
    # create the diagonal steady state matrix 
    delta_u_star = np.diag(pi)
    
    # vectorize the Laplacian matrix multiplied by each value in the vector tau
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau)):
        a_13[i] = (s_t**beta @ list_result[i]) @ (delta_u_star @ s ** alpha)
        a_31[i] = (s_t**alpha @ list_result[i]) @ (delta_u_star @ s ** beta)
        
    return a_13, a_31

## THE STEINBERG SIGNATURE ##

def numerical_area(t,t_rev):
    """
    Calculates the analytical solution for forward and reverse higher-order autocorrelation functions for a particular Laplacian matrix
    
    Parameters
    ----------
    signal : 1D array
        possible values of signal (which is a state function on the Markov process)
    L : NxN array
        column-based Laplacian matrix of linear framework graph with N vertices
    tau : 1D array
        range of intervals between values of signal taken by system
    alpha : scalar
        exponent applied to signal
    beta : scalar
        exponent applied to transpose of signal
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    """
    return np.abs(np.trapz(t)-np.trapz(t_rev))

def Laplacian_spectrum(L):
    """
    Returns the eigenvalues, left eigenvectors, and right eigenvectors (otherwise known as the spectrum) of the Laplacian matrix for a 3-vertex graph. The eigenvectors are normalized with the following conditions. 
    
    u_1 (right eigenvector of the 0 eigenvalue): normalized such that all elements sum to 1
    v_1 (left eigenvector of the 0 eigenvalue): normalized such that all elements = 1
    
    u_n, v_n (n>1): normalized such that they satisfy a bi-orthonormal condition (see below)
    
    v_i * u_j = \delta_ij (Kronecker delta function)
    
    Parameters
    ----------

    L : 3x3 array
        column-based Laplacian matrix of 3-vertex linear framework graph
        
    Returns
    -------
    
    lambda_1 : np.complex128
        0 eigenvalue (largest eigenvalue)
    
    u_1 : numpy.ndarray
        right eigenvector for the 0 eigenvalue. Also known as the steady state or invariant distribution of the Laplacian
        
    v_1 : numpy.ndarray
        left eigenvector for the 0 eigenvalue. Also known as the all ones vector
        
    lambda_2 : np.complex128
        smallest eigenvalue (largest negative eigenvalue)
    
    u_2 : numpy.ndarray
        right eigenvector for \lambda_2. Normalized such that v_2 * u_2 = 1
        
    v_2 : numpy.ndarray
        left eigenvector for the \lambda_2. Normalized such that v_2 * u_2 = 1
        
    lambda_3 : np.complex128
        middle eigenvalue
    
    u_3 : numpy.ndarray
        right eigenvector for \lambda_3. Normalized such that v_3 * u_3 = 1
        
    v_3 : numpy.ndarray
        left eigenvector for the \lambda_3. Normalized such that v_3 * u_3 = 1
    
    """
    # Define eigenvalues, left eigenvectors (v_i) and right eigenvectors (u_i)
    eigvals, left_eigvecs, right_eigvecs = scipy.linalg.eig(L, left=True, right=True)
    
    # Identify the index of the 0 eigenvalue and define it as lambda_1
    idx_1 = np.argmax(eigvals)
    lambda_1 = eigvals[idx_1]

    # Delete this eigenvalue from eigvals
    eigvals_other = np.delete(eigvals, idx_1)

    # Define the right u_1 and left v_1 eigenvectors associated with lambda_1
    u_1 = right_eigvecs[:,idx_1]
    v_1 = left_eigvecs[:,idx_1]

    # Normalize u_1 and v_1
    normalization_1 = sum(u_1)
    u_1 = u_1/normalization_1
    v_1 = v_1/v_1

    # Delete eigenvectors u_1 and v_1 from their respective arrays
    right_eigvecs_other = np.delete(right_eigvecs, idx_1, axis=1)
    left_eigvecs_other = np.delete(left_eigvecs, idx_1, axis=1)

    # Identify indexes for second and third eigenvalues
    idx_2 = np.argmin(eigvals_other)
    idx_3 = np.argmax(eigvals_other)
    
    # Define lambda_2 and lambda_3
    lambda_2 = eigvals_other[idx_2]
    lambda_3 = eigvals_other[idx_3]

    # Define left and right eigenvectors for lambda_2 and normalize appropriately
    u_2 = right_eigvecs_other[:,idx_2]
    v_2 = left_eigvecs_other[:,idx_2]
    normalization_2 = np.dot(u_2, v_2)
    u_2 = u_2/normalization_2
    v_2 = v_2/normalization_2

    # Define left and right eigenvectors for lambda_3 and normalize appropriately
    u_3 = right_eigvecs_other[:,idx_3]
    v_3 = left_eigvecs_other[:,idx_3]

    normalization_3 = np.dot(u_3, v_3)
    u_3 = u_3/normalization_3
    v_3 = v_3/normalization_3

    return lambda_1, u_1, v_1, lambda_2, u_2, v_2, lambda_3, u_3, v_3

def analytical_area(lambda_2, lambda_3, u_1, u_2, v_2, u_3, v_3, f, alpha=1, beta=3):
    """
    Compute the closed form area between higher-order autocorrelation functions (a.k.a. the Steinberg signature) for a particular 3x3 Laplacian matrix.
    
    In LaTeX, here is the area formula for the 3-vertex graph:
    
    $$ A^{\alpha,\beta} = \sum_{n>1} \dfrac{\chi^{\beta,\alpha}_n-\chi^{\alpha,\beta}_n}{\lambda_n} = \dfrac{\chi^{\beta,\alpha}_2-\chi^{\alpha,\beta}_2}{\lambda_2} + \dfrac{\chi^{\beta,\alpha}_3-\chi^{\alpha,\beta}_3}{\lambda_3}$$
    
    $$ \chi^{\alpha,\beta}_n = \left ( f^\alpha \cdot u_n \right ) \cdot \left (v_n \cdot f^{\beta *} \pi \right )$$
    
    Parameters
    ----------

    lambda_2 : np.complex128
        smallest eigenvalue (largest negative eigenvalue)
    
    lambda_3 : np.complex128
        middle eigenvalue
        
    u_1 : numpy.ndarray
        right eigenvector for the 0 eigenvalue. Also known as the steady state or invariant distribution of the Laplacian
        
    u_2 : numpy.ndarray
        right eigenvector for \lambda_2. Normalized such that v_2 * u_2 = 1
        
    v_2 : numpy.ndarray
        left eigenvector for the \lambda_2. Normalized such that v_2 * u_2 = 1
        
    u_3 : numpy.ndarray
        right eigenvector for \lambda_3. Normalized such that v_3 * u_3 = 1
        
    v_3 : numpy.ndarray
        left eigenvector for the \lambda_3. Normalized such that v_3 * u_3 = 1
    
    f : 1D array
        possible values of signal (which is a state function on the Markov process)
    
    alpha : scalar (default = 1)
        exponent applied to signal
    
    beta : scalar (default = 3)
        exponent applied to transpose of signal
        
    Returns
    -------
    
    area_ab : scalar
        the closed form area between higher order autocorrelation functions
    
    """
    
    coefficient_2ab = sum((f**alpha)*u_2) * sum(v_2*(f**beta * u_1))
    coefficient_2ba = sum((f**beta)*u_2) * sum(v_2*(f**alpha * u_1))
    
    coefficient_3ab = sum((f**alpha)*u_3) * sum(v_3*(f**beta * u_1))
    coefficient_3ba = sum((f**beta)*u_3) * sum(v_3*(f**alpha * u_1))
    
    term_1 = ((coefficient_2ba - coefficient_2ab)/lambda_2.real)
    term_2 = ((coefficient_3ba - coefficient_3ab)/lambda_3.real)
    
    area_ab = np.abs(((coefficient_2ba - coefficient_2ab)/lambda_2.real) + ((coefficient_3ba - coefficient_3ab)/lambda_3.real))
    
    return area_ab, coefficient_2ab, coefficient_2ba, coefficient_3ab, coefficient_3ba, term_1, term_2
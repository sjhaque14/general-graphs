import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from collections import Counter

# general-graph-utils.py

# This library allows the user to greate linear framework graphs (finite, directed graphs with no self-loops and labeled edges). The user can randomly generate a strongly connected and fully reversible linear framework graph G, calculate its Laplacian matrix and its spectrum, and calculate the Steinberg signature from that matrix. The user can also determine how the Steinberg signature changes as a function of increasing entropy production. This software was developed using the NetworkX software package (https://networkx.org/documentation/stable/index.html)

# Note that the user is required to create both a directed graph object and an undirected graph object. This is because some of the functions in this file require the undirected graph object as an argument (particular the cycle-related functions).

def random_graph(n):
    """
    Generates a linear framework graph that is strongly connected and fully reversible. The size of the graph is randomly determined from range (3, n), and the edges are added by randomly selecting a pair of nodes in G.
    
    Parameters
    ----------
    n : integer > 3
        the maximum number of vertices
    
    Returns
    -------
    G : NetworkX DiGraph object
        directed graph
    
    G_ud : NetworkX Graph object
        undirected graph
    """
    
    G = nx.DiGraph()
    G_ud = nx.Graph()
    
    # determine the number of nodes in G
    G_size = np.random.choice(np.arange(3,n), size=1)
    
    # add nodes to G and G_ud
    nodes = np.arange(1,G_size+1,step=1)
    G.add_nodes_from(nodes)
    G_ud.add_nodes_from(nodes)
    
    # add edges until the graph is strongly connected
    while nx.is_strongly_connected(G) == False:
        u, v = np.random.choice(nodes, size=2,replace=False)
        G.add_edge(u,v)
        G_ud.add_edge(u,v)
    
    # add edges such that the graph is fully reversible
    for edge in G.edges:
        u = edge[0]
        v = edge[1]
        if G.has_edge(v,u) == False:
            G.add_edge(v,u)
            G_ud.add_edge(v,u)
    
    return G, G_ud

def random_graph_n(n):
    """
    Generates a linear framework graph of size n that is strongly connected and fully reversible.
    
    Parameters
    ----------
    n : integer
        the size of the graph
    
    Returns
    -------
    G : NetworkX DiGraph object
        directed graph
    
    G_ud : NetworkX Graph object
        undirected graph
    """
    
    G = nx.DiGraph()
    G_ud = nx.Graph()
    
    # determine the number of nodes in G
    G_size = n
    
    # add nodes to G and G_ud
    nodes = np.arange(1,G_size+1,step=1)
    G.add_nodes_from(nodes)
    G_ud.add_nodes_from(nodes)
    
    # add edges until the graph is strongly connected
    while nx.is_strongly_connected(G) == False:
        u, v = np.random.choice(nodes, size=2,replace=False)
        G.add_edge(u,v)
        G_ud.add_edge(u,v)
    
    # add edges such that the graph is fully reversible
    for edge in G.edges:
        u = edge[0]
        v = edge[1]
        if G.has_edge(v,u) == False:
            G.add_edge(v,u)
            G_ud.add_edge(v,u)
    
    return G, G_ud

def get_nodes(G):
    """
    Returns an array of nodes in a NetworkX graph object (directed or undirected)
    
    Parameters
    ----------
    G : NetworkX DiGraph object
        directed graph
    
    Returns
    -------
    node_list : NumPy array
        list of nodes
    """
    node_list = np.array(G.nodes)
    return node_list

def get_edges(G):
    """
    Returns an array of edges in a NetworkX graph object (directed). Each edge is represented as a list [source,sink]
        
    Parameters
    ----------
    G : NetworkX DiGraph object
        directed graph
    
    Returns
    -------
    edge_list : NumPy array
        list of lists of directed edges (each directed edge is represented as a list)
    """
    edge_list = np.array(G.edges)
    return edge_list

def get_cycles(G_ud):
    """
    Returns a list of the cycles which form a basis G (must be undirected). Each element is a list of the nodes connected in a given cycle.
    
    Parameters
    ----------
    G_ud : NetworkX Graph object
        undirected graph
            
    Returns
    -------
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
    """
    
    cycle_list = [c for c in nx.cycle_basis(G_ud)]
    
    return cycle_list

def get_labels(G):
    """
    Extracts the label information for each edge in a NetworkX graph object. If G not labeled, labels are sampled as 10^x, where x is sampled uniformly between -3 and 3.
    
    Parameters
    ----------
    G : NetworkX DiGraph object
        directed graph
            
    Returns
    -------
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    label_list : 1D numpy array
        list of edge labels in G
        
    """
    
    label_dict = {}
    
    # if G weighted/labeled, extract weight information in dictionary form
    if nx.is_weighted(G)==True:
        for i in range(len(G.edges())):
            label_dict[list(G.edges())[i]] = G.get_edge_data(list(G.edges)[i][0],list(G.edges)[i][1])['weight']
    
    # if G not weighted/labeled, sample new edge label for each edge
    elif nx.is_weighted(G)==False:
        label_dict = {e: np.around(10**(np.random.uniform(-3,3, size = 1)[0]),decimals=5) for e in G.edges}
    
    # create a list of edge labels directly from the dictionary
    label_list = np.fromiter(label_dict.values(), dtype=float)
    
    return label_dict, label_list

def get_labels_ones(G):
    """
    Extracts the label information for each edge in a NetworkX graph object. If G not labeled, all labels are assigned as 1.0.
    
    Parameters
    ----------
    G : NetworkX DiGraph object
        directed graph
            
    Returns
    -------
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    label_list : 1D numpy array
        list of edge labels in G
        
    """
    
    label_dict = {}
    
    # if G weighted/labeled, extract weight information in dictionary form
    if nx.is_weighted(G)==True:
        for i in range(len(G.edges())):
            label_dict[list(G.edges())[i]] = G.get_edge_data(list(G.edges)[i][0],list(G.edges)[i][1])['weight']
    
    # if G not weighted/labeled, sample new edge label for each edge
    elif nx.is_weighted(G)==False:
        label_dict = {e: 1.0 for e in G.edges}
    
    # create a list of edge labels directly from the dictionary
    label_list = np.fromiter(label_dict.values(), dtype=float)
    
    return label_dict, label_list

def get_cycle_labels_edges(cycle_list,label_dict):
    """
    Compartmentalizes, for each cycle, the edges involved and their respective edge labels into separate data structures.
    
    Parameters
    ----------    
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
        
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    Returns
    -------
    cycle_edges_forward : list of lists
        each element is a list of the edges going around one direction of a given cycle
    
    cycle_edges_backward : list of lists
        each element is a list of the edges going around the opposite direction of a given cycle
    
    cycle_labels_forward : list of lists
        each element is a list of the labels going around one direction of a given cycle
    
    cycle_labels_backward : list of lists
        each element is a list of the labels going around the opposite direction of a given cycle
        
    """
    
    # define number of cycles
    num_cycles = len(cycle_list)
    
    # define arrays for edges in each direction
    cycle_edges_forward = [[] for i in range(num_cycles)]
    cycle_edges_backward = [[] for i in range(num_cycles)]
    
    # define arrays for labels in each direction
    cycle_labels_forward = [[] for i in range(num_cycles)]
    cycle_labels_backward = [[] for i in range(num_cycles)]

    # iterate over each cycle
    for j in range(num_cycles):
        
        # for each node in the cycle
        for i in range(1,len(cycle_list[j])):
            
            # identify what it is connected to
            source = cycle_list[j][i-1]
            sink = cycle_list[j][i]
            
            # edge (source,sink) and accompanying labels
            cycle_labels_forward[j].append(label_dict.get((source,sink)))
            cycle_edges_forward[j].append((source,sink))
            
            #edge (sink, source) and accompanying labels
            cycle_labels_backward[j].append(label_dict.get((sink,source)))
            cycle_edges_backward[j].append((sink,source))
        
        # account for the connection between the last and first elements of each cycle list
        final_source = cycle_list[j][-1]
        final_sink = cycle_list[j][0]

        # edge (final_source, final_sink) and accompanying labels
        cycle_labels_forward[j].append(label_dict.get((final_source,final_sink)))
        cycle_edges_forward[j].append((final_source,final_sink))
        
        # edge (final_sink, final source) and accompanying labels
        cycle_labels_backward[j].append(label_dict.get((final_sink,final_source)))
        cycle_edges_backward[j].append((final_sink,final_source))
        
    return cycle_edges_forward, cycle_edges_backward, cycle_labels_forward, cycle_labels_backward

def calculate_cycle_products(cycle_labels_forward,cycle_labels_backward):
    """
    Calculates the product of edge labels going in forward and reverse directions for each cycle
    
    Parameters
    ----------
    cycle_labels_forward : list of lists
        each element is a list of the labels going around one direction of a given cycle
    
    cycle_labels_backward : list of lists
        each element is a list of the labels going around the opposite direction of a given cycle
        
    Returns
    -------
    products_f : 1D array
        each element is the product of labels corresponding to the forward traversal of each cycle
    
    products_b : 1D array
        each element is the product of labels corresponding to the backward traversal of each cycle
        
    """
    products_f = np.zeros(len(cycle_labels_forward),dtype=np.float128)
    products_b = np.zeros(len(cycle_labels_backward),dtype=np.float128)

    for i in range(len(cycle_labels_forward)):
        products_f[i] = np.prod(cycle_labels_forward[i])
        products_b[i] = np.prod(cycle_labels_backward[i])
    
    return products_f, products_b

def calculate_affinities(products_f, products_b, cycle_list):
    """
    Calculates the cycle affinity (e.g. thermodynamic force) for each cycle in a graph
    
    Parameters
    ----------
    products_f : 1D array
        each element is the product of labels corresponding to the forward traversal of each cycle
    
    products_b : 1D array
        each element is the product of labels corresponding to the backward traversal of each cycle
        
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
        
    Returns
    -------
    
    total_affinities : 1D array
        each element is the thermodynamic force for each cycle in the graph, corresponding to their order in cycle_list
    
    """
    
    num_cycles = len(cycle_list)
    
    total_affinities = np.zeros(num_cycles,dtype=np.float128)
    
    for i in range(num_cycles):
        total_affinities[i] = np.log(products_f[i]/products_b[i])
    
    return total_affinities

def shared_edges_cycles(cycle_list, cycle_edges_forward, cycle_edges_backward):
    """
    Returns a list of all edges that are mutual to more than one cycle in G
    
    Parameters
    ----------
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
        
    cycle_edges_forward : list of lists
        each element is a list of the edges going around one direction of a given cycle
    
    cycle_edges_backward : list of lists
        each element is a list of the edges going around the opposite direction of a given cycle
        
    Returns
    ----------
    shared_cycle_edges_list : list
        list of all the pairs of reversible edges that are shared between at least 2 cycles in G
    
    all_cycle_edges_forward : list
        list of all the edge tuples that are recorded in cycle_edges_forward
    
    """
    all_cycle_edges = []
    all_cycle_edges_forward = []

    for i in range(len(cycle_list)):
        for j in range(len(cycle_list[i])):
            all_cycle_edges.append(cycle_edges_forward[i][j])
            all_cycle_edges.append(cycle_edges_backward[i][j])
            all_cycle_edges_forward.append(cycle_edges_forward[i][j])
    
    shared_cycle_edges_dict = Counter(all_cycle_edges)
    
    shared_cycle_edges_list = [edge for edge, count in shared_cycle_edges_dict.items() if count >= 2]
    
    return shared_cycle_edges_list,all_cycle_edges_forward

def equilibrium_params_2(cycle_list,cycle_edges_forward,shared_cycle_edges_list,all_cycle_edges_forward,cycle_labels_forward,products_f,products_b):
    """
    Calculates the cycle affinity (e.g. thermodynamic force) for each cycle in a graph
    
    Parameters
    ----------
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
        
    cycle_edges_forward : list of lists
        each element is a list of the edges going around one direction of a given cycle
    
    shared_cycle_edges_list : list
        list of all the pairs of reversible edges that are shared between at least 2 cycles in G
    
    all_cycle_edges_forward : list
        list of all the edge tuples that are recorded in cycle_edges_forward
        
    cycle_labels_forward : list of lists
        each element is a list of the labels going around one direction of a given cycle
    
    products_f : 1D array
        each element is the product of labels corresponding to the forward traversal of each cycle
    
    products_b : 1D array
        each element is the product of labels corresponding to the backward traversal of each cycle
        
    Returns
    -------
    
    cycle_labels_forward : list of lists
        updated cycle_labels_forward with new edge labels
        
    edge_tracker: list
        list of edges with altered labels
        
    index_tracker: list
        list of second indices of altered edges in cycle_labels_forward
    
    """
    num_cycles = len(cycle_list)
    # tracking edges that have had their values altered
    edge_tracker = []
    index_tracker = []
    # count the cycles that have been initialized at equilibrium
    cycles_done = -1
    
    # remove all shared edges from cycle_edges_forward
    a = set(all_cycle_edges_forward)
    b = set(shared_cycle_edges_list)
    new_cycle_edges_forward = list(a.difference(b))        
    
    for i in range(num_cycles):
        
        while cycles_done < i:
            # choose a random edge in the "forward" direction
            j = np.random.randint(len(cycle_list[i]),size=1)[0]
            edge = cycle_edges_forward[i][j]
            edge_label = cycle_labels_forward[i][j]
            
            # if the edge is not shared with other cycles, recalculate its value
            if ((edge in new_cycle_edges_forward)==True):
                #print(j)
                # recalculate edge label using the cycle condition
                edge_label = 1/(products_f[i]/(edge_label*products_b[i]))
                cycle_labels_forward[i][j] = edge_label
                # add that edge to edge_tracker
                edge_tracker.append(cycle_edges_forward[i][j])
                index_tracker.append(j)
                cycles_done += 1
            else:
                continue
            
    return cycle_labels_forward, edge_tracker, index_tracker

def reformat_labels(cycle_list, cycle_edges_forward, cycle_labels_forward, edge_tracker, index_tracker, label_dict, label_list):
    """
    Changes the parameterization of a NetworkX graph object from its initial parameterization
    
    Parameters
    ----------
    
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
    
    cycle_edges_forward : list of lists
        each element is a list of the edges going around one direction of a given cycle
    
    cycle_labels_forward : list of lists
        updated with new values for certain edges
        
    edge_tracker : list of lists
        list of edges with labels that were changed to initialize the system in an equilibrium steady state
    
    index_tracker: list
        list of second indices of altered edges in cycle_labels_forward
        
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    label_list : 1D numpy array
        list of edge labels in G
    
    Returns
    -------
    
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels (updated)
        
    label_list : 1D numpy array
        list of edge labels in G (updated)
    """
    
    num_cycles = len(cycle_list)
    
    for i in range(num_cycles):
        #print(cycle_edges_forward[i][index_tracker[i]])
        label_dict[edge_tracker[i]] = cycle_labels_forward[i][index_tracker[i]]
        
    label_list = np.fromiter(label_dict.values(), dtype=float)
    
    return label_dict, label_list

def Laplacian_all(edge_list,label_list,node_list):
    """
    Calculates the column-based Laplacian matrix for any graph. The entries of the Laplacian are computed using the following mathematical formula:
    
    L_{ij}(G) = e_{ij} if i \neq j
    L_{ij}(G) = -\sum_{v \neq j} e_{vj} if i = j.
    
    Parameters
    ----------
    edge_list : 1D array
        list of each edge in the graph object G, each element is a tuple (source,sink)
    
    label_list : 1D array
        list of edge labels in the graph
        
    node_list : 1D array
        list of nodes in the graph
    
    Returns
    -------
    
    L : num_nodes x num_nodes array
        the Laplacian matrix of the graph G
        
    """
    
    num_nodes = len(node_list)
    num_edges = len(edge_list)
    
    L = np.zeros(shape=(num_nodes,num_nodes),dtype=np.float128)
    
    # off-diagonal entries
    for x in range(num_edges):
        k = np.around(edge_list[x][0]-1,decimals=5)
        j = np.around(edge_list[x][1]-1,decimals=5)

        L[k,j] = label_list[x]
    
    # diagonal entries
    sums = np.around(-1*np.sum(L,axis=0), decimals=5)

    for i in range(num_nodes):
        L[i,i] = sums[i]
    
    return L

def steady_state_spectrum(L):
    """
    Calculates the steady-state distribution for the any linear framework graph by computing the right eigenvector associated with eigenvalue 0 and normalizing it by the sum of all entries
    
    Parameters
    ----------
    L : num_nodes x num_nodes array
        the Laplacian matrix of the graph G
    
    Returns
    -------
    pi : 1D array
         the steady state distribution for a 3-vertex graph K.
    
    """
    
    eigvals, eigvecs = scipy.linalg.eig(L)
    pi_all = eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)
    
    return pi_all

def G_duplicate_data_structures(G,node_list,edge_list,label_list,label_dict,size,L):
    """
    Creates new data structures for a graph G, in case you want to manipulate a graph's topology
    """
    
    node_list_G = node_list
    edge_list_G = edge_list
    label_list_G = label_list
    label_dict_G = label_dict
    size_G = size
    L_G = L
    
    return node_list_G, edge_list_G, label_list_G, label_dict_G, size_G, L_G

def assign_labels(G, label_dict):
    """
    Adds labels to a graph object
    
    Parameters
    ----------
    G : NetworkX graph object (directed)
    
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    Returns
    -------
    
    G : with labels
        
    """
    for e in G.edges:
        G[e[0]][e[1]]['weight'] = label_dict[e]
        
    return G

def make_observable(node_list):
    """
    Create the observable vector f for a graph with size num_nodes = len(node_list). The observable vector is a function on the states of the Markov process defined for the linear framework graph: when the system exists in state k, f takes vaue f_k.
    
    Parameters
    ----------
    node_list : 1D array
        list of nodes in the graph
    
    Returns
    -------
    f : 1D array
        list of values that the observable f assumes based on the state the Markov process exists in at a given time t 
    
    """
    
    num_nodes = len(node_list)
    
    f = np.zeros(num_nodes,dtype=np.float128)
    
    for i in range(0,num_nodes):
        f[i] = 3+(2*i)
    
    return f

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
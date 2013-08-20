'''
Created on 15 Aug 2013

@author: Gavin Smith
@organization: Horizon Digital Economy Research, The University of Nottingham.
'''

from __future__ import division
import platform
if platform.node() == 'gavin3':
    import libQuantization_gavin3 as libQuantization
else:
    import libQuantization as libQuantization #@UnresolvedImport

import numpy as np
from scipy.cluster.vq import vq
import logging


def sanity( ts_set, quant_fn ):
 
    """
    Given a quantisation function and a set of time series, computes the objective function score directly from the time series themselves.
    """
 
    score = 0
    ct = 0
    for ts_whole in ts_set:
        for ts_part in ts_whole:
            bit = abs( ts_part[1] - quant_fn(ts_part[0]) )
            score += bit
            ct += 1
    score /= ct 
  
    return score


def sanity2( ts_set, quant_fn, prob_matrix, e_syms, f_syms ):

    """
    Given a quantisation function and a set of time series, computes the objective function score using a precomputed probability matrix.
    
    Args:
        e_syms (list): all possible symbols for the behavioural space sorted in ascending order MUST be equi-spaced.
        f_syms (list): all possible symbols for the covariate space, sorted in ascending order. MUST be equi-spaced.
    
    """

    score = 0
    for e_idx in range( len(e_syms) ):
        for f_idx in range( len(f_syms) ):
            bit = prob_matrix[e_idx*len(f_syms)+f_idx] * abs( f_syms[f_idx] - quant_fn( e_syms[e_idx] ) )
            score += bit
    
    return score

def mine_prob( orig_ts_XbyTS_LENby2_as_indices_into_e_f, e, f, gpu_id ):
    """
    Computes the joint probability matrix between the behavioural and co-variate variables. 
    Computes the cumulative objective function value cache, keeping only the cumulative objective values for f = \inf for all e.
    
    Args:
        orig_ts_XbyTS_LENby2_as_indices_into_e_f: list of time series, with each time series being a list of pairs of [behavioural index, co-variate index]
                                                    where each index corresponds to a value in the respective e and f arrays. Should be constructed via a 
                                                    call to the function convert_to_index_representation.
        e (list): all possible symbols for the behavioural space sorted in ascending order MUST be equi-spaced.
        f (list): all possible symbols for the covariate space, sorted in ascending order. MUST be equi-spaced.
        gpu_id (int): the gpu_id to run the GPU parts on. Found via ~/NVIDIA_CUDA-5.0_Samples/bin/linux/release/deviceQuery
    
    Returns:
        cumulative_objective_array (numpy float64 array, 1D): The final column of each e (row) vs f (column) cumulative matrix for each slice, 
                                                                where a slice corresponds to one of f [Q(e)] values. The columns as flatterned
                                                                into a 1D array by concatenating each column in slice index order. The cumulative
                                                                matrix is not the cumulative probabilities but the cumulative objective values.
                                                                
        probability_matrix (numpy float64 array 1D): The flatterned joint P(e,f) probability matrix where rows refer to e, and columns refer to f.
                                                     The matrix is flatterned by concatenating rows.
        
    Description:
    TODO


    """
 
    # Construct the data structures required for the GPU version

    print 'Building p(a,b)a and p(a,b) cumulative matrices.....'


    prob_e_f_matrix = np.zeros( len(e)*len(f), dtype = np.float64, order = 'C' )


    print 'Mining the probabilities'
    
    for ts in orig_ts_XbyTS_LENby2_as_indices_into_e_f:
        for pt in ts:
            e_idx = pt[0]
            f_idx = pt[1]
                
            idx = (len(f) * e_idx) + f_idx
            
            prob_e_f_matrix[idx] += 1

    prob_e_f_matrix = prob_e_f_matrix / sum(prob_e_f_matrix)
    
    
    ########################
    
    
    if not len( prob_e_f_matrix[prob_e_f_matrix>1] ) == 0:
        raise Exception('Mined probability of greater than 1: {}'.format(prob_e_f_matrix[prob_e_f_matrix>1]))

    print '\nConstructing the cache'
    dest2 = np.zeros( len(f)*len(e), dtype = np.float64, order = 'C') # where len(f) * len(e) == num_cols * num_slices 

    
    libQuantization.GPU_GET_ASK_CACHE( np.int32(gpu_id), prob_e_f_matrix, np.int32( len(f) ), np.float64( abs(f[1]-f[0]) ), dest2  )


    return dest2, prob_e_f_matrix



def convert_to_index_representation( orig_ts_set, e, f ):
    """
    Args: 
        orig_ts_set (list of time series, each time series a list of [behavioural value, covariate value] pairs): the original time series list. This is a list of time series of which each time series is a list of [behavioural,covariate] point pairs
        e (list): all possible symbols for the behavioural space sorted in ascending order MUST be equi-spaced.
        f (list): all possible symbols for the covariate space, sorted in ascending order. MUST be equi-spaced.
    
    returns:
        time series set of the same dimensions as orig_ts_set, but with each [behvaioural value, covariate value] pair replaced with and index into
        the respective e and f sets.
    """
    
    converted_ts = []
    for ts in orig_ts_set:
        n_ts = []
        for pt in ts:
            n_ts.append( [ int(vq( np.asarray([pt[0]]), np.asarray(e) )[0]), int(vq( np.asarray([pt[1]]), np.asarray(f) )[0]) ] )
    
        converted_ts.append( n_ts )
    
    return converted_ts


def perform_clustering(  orig_ts_set, e_syms, f_syms, num_syms, gpu_id = 0 ):
    """
    Args:
        orig_ts_set (list of time series, each time series a list of [behavioural value, covariate value] pairs): the original time series list. This is a list of time series of which each time series is a list of [behavioural,covariate] point pairs
    
        e_syms (list): all possible symbols for the behavioural space sorted in ascending order MUST be equi-spaced.
    
        f_syms (list): all possible symbols for the covariate space, sorted in ascending order. MUST be equi-spaced.
    
        num_syms (int): the number of symbols to quantise to.
    
        gpu_id (int): the gpu_id to run the GPU parts on. Found via ~/NVIDIA_CUDA-5.0_Samples/bin/linux/release/deviceQuery
    
    Returns: the results, the joint probability matrix and the cumulative probability matrix. The last two are for debugging.
    
    results: A dict containing: 
    `        results['map_fn']: a function that takes one parameter (a value in the behavioural space) and returns the corresponding covariate value as determined by the learnt quantiser.
             results['fn_score']: the minimum value of the objective function for the learnt quantiser
             results['center_pts']: the center points of the learnt quantiser
             results['breakpoints']: the breakpoints of the learnt quantiser
    
    """
    
    ts_as_idxs = convert_to_index_representation(orig_ts_set, e_syms, f_syms)
    
    cumulative_prob_matrix, prob_e_f_matrix = mine_prob( ts_as_idxs, e_syms, f_syms, gpu_id )
    
    results = CLUSTER_GPU( gpu_id, e_syms, f_syms, num_syms, 199, cumulative_prob_matrix)

    return results, prob_e_f_matrix, cumulative_prob_matrix

def CLUSTER_GPU( gpu_id, s_e_space, s_f_space, q_levels, method, cum_slices):
    """
    Args:
        gpu_id (int): the gpu_id to run the GPU parts on. Found via ~/NVIDIA_CUDA-5.0_Samples/bin/linux/release/deviceQuery
    
        s_e_space (list): all possible symbols for the behavioural space sorted in ascending order MUST be equispaced.
    
        s_f_space (list): all possible symbols for the covariate space, sorted in ascending order. MUST be equispaced.
    
        q_levels (int): the number of symbols to quantise to.
    
        method (int): currently always 199
    
        cum_slices (numpy nd-array): cumulative slices matrix as constructed by the mine_prob method
    
    returns: A dict containing: 
    `        results['map_fn']: a function that takes one parameter (a value in the behavioural space) and returns the corresponding covariate value as determined by the learnt quantiser.
             results['fn_score']: the minimum value of the objective function for the learnt quantiser
             results['center_pts']: the center points of the learnt quantiser
             results['breakpoints']: the breakpoints of the learnt quantiser
    """
    
    print 'Calling CLUSTER_GPU'

    
    gpu_results = libQuantization.GPU_cluster(np.int32(gpu_id), np.int32(len(s_e_space)),np.int32(len(s_f_space)), cum_slices,np.int32(q_levels),np.int32(method))
   

    fn_score = gpu_results[0]
    
    breakpoints = gpu_results[1:q_levels]
    
    gpu_breakpoint_means = gpu_results[q_levels:]

    
    real_breakpoints =  [ s_e_space[x] for x in breakpoints ]
    
    real_breakpoints.append(np.Inf)
    
    real_breakpoints = np.asarray(real_breakpoints)

    center_pts = np.asarray( [ s_f_space[x] for x in gpu_breakpoint_means ] )
    
    print 'Center points: {}'.format( center_pts )
    print 'Breakpoints: {}'.format( real_breakpoints[:-1] ) # we drop off the +Inf from the breakpoints for printing as this is only used in the quantisation (lambda) function
        
    return {'map_fn': lambda v: center_pts[ np.searchsorted(real_breakpoints, v, side = 'left')], 'fn_score': fn_score, 'result': gpu_results[1:], 'center_pts':center_pts, 'breakpoints':real_breakpoints[:-1]  }
    




 
def test1():   
    """
    A very basic test case. No automatic solution checking coded.
    """ 
    
    orig_ts = [ 
               [ [1,1] ,[1,1],[2,2],[2,2] ],
               [ [1,1] ,[1,1],[2,2],[2,2] ]
               #[ [6,6] ,[5,5],[4,4],[3,3] ]
               ]
    
    e_syms = [1,2,3]#,4,5,6,7,8,9,10]
    
    f_syms = [1,2]#,3,4,5,6,7]
    
    num_syms = 2
    
    # Don't know your gpu_id ? Run the following from a terminal:
    # ~/NVIDIA_CUDA-5.0_Samples/bin/linux/release/deviceQuery
    
    perform_clustering( orig_ts, e_syms, f_syms, num_syms, gpu_id = 0 )
    

def test2():
    """
    A basic test case and solution coded in.
    """
    orig_ts = [ [ [1,2] ,[2,2],[7,4],[8,2],[9,6],[10,6] ] ]

    e_syms = [1,2,3,4,5,6,7,8,9,10]
    f_syms = [2,3,4,5,6]
    num_syms = 2
    r, prob_e_f_matrix, cum_slices = perform_clustering( orig_ts, e_syms, f_syms, num_syms, gpu_id = 0 )
    
    #print 'sanity: {}, sanity2: {}'.format( sanity( orig_ts, r['map_fn']) , sanity2( orig_ts, r['map_fn'], prob_e_f_matrix, e_syms, f_syms ) )
    
    #correct_sol = np.ascontiguousarray([2,9,2,3,6], dtype=np.int32) # this is incorrect since it has to be translated into indexes, not raw values
    correct_sol = np.ascontiguousarray([1,8,0,1,4], dtype=np.int32)
    alternate_eval = libQuantization.GPU_single_eval(np.int32(len(e_syms)),np.int32(len(f_syms)), cum_slices,np.int32(num_syms),np.int32(199), correct_sol)
    

    
    center_pts = [2,3,6]
    real_breakpoints = np.asarray([2,9])
    
    print ''
    s1 = sanity( orig_ts, r['map_fn'])
    if abs( s1 - r['fn_score']) > 0.000000001:
        raise Exception( 'Error: Found answer has a CPU score: {} but GPU evaluated it to a score of {}'.format( s1 , r['fn_score'] ) )
    
    s2 = sanity( orig_ts, lambda v: center_pts[ np.searchsorted(real_breakpoints, v, side = 'left')] )
    
    if abs( s2 - alternate_eval) > 0.000000001:
        raise Exception( 'Alternative evaluation answer has CPU score: {} but GPU evaluated it to a score of {}'.format( s2 , alternate_eval ) )
    
    print '\nTest passed. \nFound answer has a CPU score: {} and GPU also evaluated it to a score of {}\n'.format( s1 , r['fn_score'] )



if __name__ == '__main__':   
    test2()    
    



'''
Created on 19 Aug 2013

@author: gavin
'''

import GPUCluster
import numpy as np
import sqlite3
from copy import copy


if __name__ == '__main__':
    
    e_base_uniform_quant = 1000
    
    num_syms = 2
    
    conn = sqlite3.connect('/home/gavin/safe/data/CERDatabase/cer_electric.sqlite')
    
    curs = conn.cursor()
    
    curs.execute( """
                    SELECT s.meter_id, 
                       s.avg avg_30m, s.var var_30m, s.skew skew_30m, s.max max_30m,
                       d.avg avg_day, d.var var_day, d.skew skew_day, d.max max_day 
                    FROM cer_electric_30m_summary_res s, cer_electric_day_summary_res d
                    WHERE s.meter_id = d.meter_id
                    ORDER by s.meter_id;
                    """ )
    
    behavioural_data = np.asarray( curs.fetchall(), dtype=np.float32 );
    meter_ids = copy( behavioural_data[:,0] );


    curs.execute( """
        SELECT meter_id, "{}"
        FROM cer_survey_res 
        ORDER by meter_id;
    """.format('453') )
    
    # could use float64
    covariate_data = np.asarray( curs.fetchall(), dtype=np.float32 );
    
    # get a binary mask over the values that actually exist
    actual_idxs = np.isfinite(covariate_data[:,1])
    
    e_data = behavioural_data[actual_idxs,1] # avg_30min
    f_data = covariate_data[actual_idxs,1]
    
    ts = zip( e_data, f_data )
    
    ts_set = [ts]
    
    
    # must be an equi-spaced uniform quantisation 
    e_syms = np.linspace(np.min(e_data), np.max(e_data), num = e_base_uniform_quant, endpoint = True)
    
    # must be an equi-spaced uniform quantisation
    f_syms = range(int(np.min(f_data)-1), int(np.max(f_data)+1))
    
    
    """Returns: the results, the joint probability matrix and the cumulative probability matrix. The last two are for debugging.
        results: A dict containting: 
    `        results['map_fn']: a function that takes one parameter (a value in the behavioural space) and returns the corresponding covariate value as determined by the learnt quantiser.
             results['fn_score']: the minimum value of the objective function for the learnt quantiser
             results['center_pts']: the center points of the learnt quantiser
             results['breakpoints']: the breakpoints of the learnt quantiser
    """
    r = GPUCluster.perform_clustering( ts_set, e_syms, f_syms, num_syms, gpu_id = 0 )
    
    
    
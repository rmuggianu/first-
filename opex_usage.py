# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import argparse
import math
import pandas as pd
import numpy as np

import json

from opex import Opex 
from walk import TreeMatrix


# =============================================================================
# This is the start
# =============================================================================
def main ():
    """ main function definition"""   
    NUMBER_DAYS_PER_MONTH = 30
    
    cmdline_inputs = argparse.ArgumentParser()
    cmdline_inputs.add_argument("-c", "--config", required=True, \
                                help="Path for the config file")
    args = vars(cmdline_inputs.parse_args())
# =============================================================================
# config file contains files to be processed. At the moment it is used only to 
# assemble an output file from from all the input files and the requirements 
# listed    
# =============================================================================
    config_file = args["config"]
    
    with open(config_file) as json_file:
        initializator = json.load(json_file)

# =============================================================================
# Derive from the configuration file the # of months upon which performing the
# evaluation. If not present a default value is used.
# =============================================================================
    default_observed_time = 24
    if 'T' in initializator :
        observed_time = initializator['T']
    else:
        observed_time = default_observed_time
# =============================================================================
# extract the number of days per month
# =============================================================================

    default_days_per_month = NUMBER_DAYS_PER_MONTH
    if 'Monthly Days' in initializator :
        days_per_month = initializator['Monthly Days']
    else:
        days_per_month = default_days_per_month

# =============================================================================
# extract the number of cumulated licenses used for the simulation
# =============================================================================
    default_cumulated_licenses = 0
    if 'Cumulated License' in initializator :
        cumulated_licenses = initializator['Cumulated License']
    else:
        cumulated_licenses = default_cumulated_licenses

   
# =============================================================================
# extract the rule set for the maintenance calculation from the file defined 
# in json configuration file
# =============================================================================     
    default_rules_file = "./rules.txt"
    if 'Rules' in initializator :
        rules = initializator['Rules']
    else:
        rules =  default_rules_file
    
    rules_set = list()
    for file in rules:
        with open(file) as rules_file:
            rules_set.append(pd.read_csv(
                    file,';',header=0,
                    names= ["cost", "<= t","t<",
                            "<=Licenses", "Licenses<"]))
# =============================================================================
# output file for licenses time series and overall cost
# =============================================================================
    default_cost_file = "./cost_table_.txt"
    if 'Cost Result' in initializator :
        cost_results = initializator['Cost Result']
    else:
        cost_results = default_cost_file 


    default_output_file = "./license_profile_out.txt"
    if 'License Profile' in initializator:
        output_file = initializator['License Profile']
    else:
        output_file = default_output_file
# =============================================================================
#  A set of license profile is generated, with granularity of 1 month
# 
# =============================================================================
    
    path = np.zeros(observed_time+1)
    
    tree = TreeMatrix()
    steps = observed_time+1
# the logic here is to use the average value ...
    average_value_step = cumulated_licenses/steps
    jumps = math.ceil((average_value_step)*2)+1
    all_path = np.ones(steps, dtype = np.int64 )*average_value_step
    tree.setTree(steps, jumps, 1)

    
    for i in range(tree.getTreeNodeNextNumber()) :
        all_path = tree.computeRandomPath(
                    step=0, 
                    cumulated_license = cumulated_licenses,
                    jump=i,path=path, 
                    path_matrix=all_path)
        
    all_path = tree.computeDiracPath(steps,
                                     cumulated_license = cumulated_licenses,
                                     path = path,
                                     path_matrix = all_path)
    

    all_path = \
    tree.computeDiracPathVector(steps,
                                cumulated_license =cumulated_licenses,
                                iteration_number = 100,
                                path = path,
                                path_matrix = all_path)
            
# =============================================================================
# Now evaluate the opex for all the licenses time series generated
# =============================================================================       
        
    number_of_time_series = all_path.shape[0]
    time =  np.linspace(0,observed_time, steps, dtype=np.int64) \
    *days_per_month
    with open(cost_results,"w") as cost_results:  
        time.tofile(cost_results,sep = ",", format = "%d")
        print(",", "opex cost","\n", file=cost_results)
        for path in range(number_of_time_series):
            time_series_array = {"time_in_day": time, 
                                 "Purchased_Licenses": all_path[path]}
            license_purchase_history = \
                pd.DataFrame(time_series_array,
                             index= time)
# the idea is to cycle over different rules set and to write the corresponding
# in each of the column. to do that in the initialization file we need to 
# insert list and read them.
            tot_stack=np.ones(0)
            for rules in rules_set:
                opex = Opex(0) 
                opex.initialise(observed_time, output_file,
                                rules, license_purchase_history)
                opex.calculate()
                total = np.ones(1)*opex.getOpexTotal()
                tot_stack = np.hstack((tot_stack,total))
            norm = all_path[path]
            product = np.vdot(norm, time)/(cumulated_licenses)
            cum_stack= np.hstack((all_path[path].cumsum(),product, 
                                  tot_stack))
            cum_stack[steps-1] = cumulated_licenses
            cum_stack.tofile(cost_results, sep = ",")
            print("", file=cost_results)

# =============================================================================
    
    return

if __name__ == "__main__":
    main()


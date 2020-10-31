# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

def GetUnitaryCost(time, CumulatedLicense, RuleSet):
    """Given the defined set of rule, return the unitary opex cost for license"""  

    UnitaryCost = 0
    index = RuleSet[(RuleSet['<= t']<= time) & (time < RuleSet['t<']) &
               (RuleSet['<=Licenses'] <= CumulatedLicense) &
               (CumulatedLicense < RuleSet['Licenses<'])].index[0]
#   Note that UC is a series whose index is the one from the original DataFrame
#   that is why it is needed to find that index in order to be able to return
#   the right unitary cost value
    UnitaryCost = RuleSet['cost'][index]
    return UnitaryCost


def GetThresholdforCumulatedLicenseRule(time, CumulatedLicense, PurchasedLicense,
                                        RuleSet):
    """returns either 0 or the cumulated license threshold reached """  

# Observe there might be rules always true which need to be discarded
    cumulated_licenses_low = CumulatedLicense-PurchasedLicense
    hit_rule_low_index = RuleSet[(RuleSet['<=Licenses'] <= cumulated_licenses_low) &
        (cumulated_licenses_low < RuleSet['Licenses<']) &
        (RuleSet['<= t']<= time) & (time < RuleSet['t<'])].index[0]
        
    cumulated_licenses_high = CumulatedLicense 
    hit_rule_high_index = (RuleSet[(RuleSet['<=Licenses'] <= 
                                    cumulated_licenses_high) & 
    (cumulated_licenses_high < RuleSet['Licenses<']) &
    (RuleSet['<= t']<= time) & (time < RuleSet['t<'])].index[0])
    
    if (RuleSet['<=Licenses'][hit_rule_high_index] > 
        RuleSet['<=Licenses'][hit_rule_low_index]):
        
        threshold =  RuleSet['<=Licenses'][hit_rule_high_index]
        
    else:
        
        threshold = 0
        
    return threshold

def GetHitRule(time, CumulatedLicense, RuleSet):
    """return the rules that is being hit"""  
    
    HitRule = RuleSet[(RuleSet['<= t']<= time) & (time < RuleSet['t<']) &
               (RuleSet['<=Licenses'] <= CumulatedLicense) &
               (CumulatedLicense < RuleSet['Licenses<'])]

    return HitRule

def calculateOpexCoreAlgorithm(cost_table = pd.DataFrame(),
                               total_opex = 0,
                               days_in_month = 30,
                               cap = np.inf,
                               algorithm_type = "memory_less"):
    time = cost_table.index
    time2 = cost_table.columns
    y = len(time)
    x = len(time2)
    cum_opex = total_opex
    for rows in range(y):
        for columns in range(x):
            if ( (type(time2[columns]) is not str) and
                (time2[columns]> time[rows]) ):
                cost_table.iloc[rows,columns]= (
                        cost_table.iloc[rows,0]*
                        cost_table.iloc[rows,1]*
                        (time2[columns]-time2[columns-1])/days_in_month)
                cum_opex += cost_table.iloc[rows, columns] 
    return cum_opex

class Opex:
    """Class definition for cumulated opex"""
    def __init__(self, initial_opex):
        self.opex = initial_opex
        
    def initialise(self, T, output="./cost.csv", 
                   rules=pd.DataFrame(), licenses=pd.DataFrame()):
        """initialize the object with time, rules and licenses"""
#        print("this is the initialize method")
#        print("value of T is", T)
        self.T = T
        self.cost_t_s = output 
        self.rule_set=rules
# =============================================================================
# time cannot be greate than T  --> licenses[(licenses['<= t']<= T)]
# =============================================================================
        self.purchased_license=licenses[(licenses.index<= T*30)]
        
# =============================================================================
# It returns the cumulated amount of opex. 
# =============================================================================
    def getOpexTotal(self):
        return int(self.opex)
  
# =============================================================================
# get and set utilities
# =============================================================================
    def getObservedTime(self):
       return self.T
   
    def setObservedTime(self, observedTime=0):
        self.T = observedTime
        
    def getRulesSet(self):
        return self.rule_set
    
    def setRulesSet(self, rules = pd.DataFrame()):
        self.rule_set = rules
        
    def getLicenseProfile(self):
        return self.purchased_license
    
    def setLicenseProfile(self, licenses = pd.DataFrame()):
        self.purchased_license = licenses
    
# =============================================================================
# main opex evaluation algorithm
# ============================================================================  

    def calculate(self):
        """based on the input given the cumulated opex costs are calculated"""
#        print("this is the calculate method")       
    
    # derive the cost unitary matrix.      

# =============================================================================
# Those list contains information related to the purchase license profile
# index contains the time indication
# delta contains the increment in the licenses purchased
# unitary cost contains the maintenance cost associated to delta
# =============================================================================
# =============================================================================
# initialize variable
# =============================================================================
# does it make sense to consider numpy as array, no ?
        cumulated_licenses = 0    
        list_index=[]
        list_delta=[]
        list_unitary_cost=[]

# =============================================================================
# index contains the tj when licenses are purchased       
# =============================================================================
        for index in self.purchased_license.index:
            
            purchased_licenses=(self.purchased_license
                                ['Purchased_Licenses'][index])
            cumulated_licenses= (cumulated_licenses + purchased_licenses)
            
            unitary_cost= GetUnitaryCost(index, 
                              cumulated_licenses,
                              self.rule_set)
# =============================================================================
# this is the scenario where newly purchased license go beyond a threshold 
# set by a cumulated license rule
# =============================================================================

            lower_threshold = GetThresholdforCumulatedLicenseRule(
                    index,
                    cumulated_licenses,
                    purchased_licenses,
                    self.rule_set)
# =============================================================================
# if lower_threshold is positive it means that the cumulated licenses purchased
# crossed a unitary cost discontinuity, therefore the license purchased at some
# tj (delta) belongs to 2 different unitary cost 
# =============================================================================


            if lower_threshold > 0:
                
                delta = cumulated_licenses - lower_threshold + 1
                previous_unitary_cost= (list_unitary_cost[
                   len(list_unitary_cost)-1])
                list_index.append(index)
                list_delta.append(purchased_licenses-delta)
                list_unitary_cost.append(previous_unitary_cost)
                list_index.append(index)
                list_delta.append(delta)
                list_unitary_cost.append(unitary_cost)
                
            else:
                
                list_index.append(index)
                list_delta.append(purchased_licenses)
                list_unitary_cost.append(unitary_cost)
# =============================================================================
#  unitary cost matrix create a tabular format from the list used above 
#  observe that on the column it would a  mistake to add the point in time
#  where the cumulated license crosses a rule discontinuity.
#  the reshape function transform an array into a column vector.                
# =============================================================================
 
      
        length=len(list_delta)
        dedup_list= list(dict.fromkeys(list_index))
        cost_matrix = pd.DataFrame(np.concatenate(
                    (np.array(list_delta).reshape(length,1),
                     np.array(list_unitary_cost).reshape(length,1),
                     np.zeros((length,len(dedup_list)))),axis=1),
                     index=list_index,
                     columns=['purchased licenses']+['unitary cost']
                     +dedup_list)
          
# =============================================================================
# this algorithm assumes that once a license is associated with a maintenance 
# cost it will be unchanged over time with only the newly purchased license
# being associated with new costs.
# the unitary cost in is column 1, the corresponding number of license is in 
# column 0. 
# in case the unitary cost is not kept constant by row, it must be searched
# for in the rows using the value in the column to find the right unitary cost
# ... if column is 700, that value shall be searched in the index (rows)
# cost_matrix['unitary cost][time2[colums]]
# =============================================================================
#        time = cost_matrix.index
#        time2 = cost_matrix.columns
#        y = len(time)
#        x = len(time2)
#        total_opex = self.opex
#        for rows in range(y):
#            for columns in range(x):
#                if ( (type(time2[columns]) is not str) and
#                    (time2[columns]> time[rows]) ):
#                    cost_matrix.iloc[rows,columns]= (
#                            cost_matrix.iloc[rows,0]*
#                            cost_matrix.iloc[rows,1]*
#                            (time2[columns]-time2[columns-1])/30)
#                    total_opex += cost_matrix.iloc[rows, columns] 
        self.opex = calculateOpexCoreAlgorithm(cost_table = cost_matrix,
                                               total_opex = self.opex)
        self.cost_t_s = cost_matrix
        
#    print("end")

    def plot(self, output_file):
        print("this is the plot method")
        self.cost_t_s.to_csv(output_file)
        pd_graph = self.cost_t_s
#=============================================================================
# pay attention not to add the columns containing the purchased
# licenses and unitary cost 
#=============================================================================
        drawing = plt.figure(figsize=(12,8), dpi=400, constrained_layout=True)
        grid = GridSpec(10, 8, figure=drawing)
        x_values = self.purchased_license.index
# =============================================================================
# drawing cumulated purchased licenses over time
# =============================================================================
        ax_license = drawing.add_subplot(grid[0:3,0:3])
        
        ax_license.set_title('purchased licenses', {'fontsize': 'x-small'})

        
        ax_license.step(x_values,
                        self.purchased_license['Purchased_Licenses']
                         .cumsum(), where='post')
# =============================================================================
#  just formatting, to be replaced with a function                            
# =============================================================================
        xlim = math.ceil(self.purchased_license.index.max())
        xticks = list(range(0,xlim,180))
        xticks.append(xlim)

        ax_license.set_xticks(xticks)
        ax_license.set_xticklabels(xticks, {'fontsize': 'xx-small',
                                            'rotation': 45})
        
        ytick_list = ax_license.get_yticks()
        ymax = math.ceil(self.purchased_license['Purchased_Licenses'].max())
        ymin = math.ceil(self.purchased_license['Purchased_Licenses'].min())
        ymean= math.ceil(self.purchased_license['Purchased_Licenses'].mean())
        ymedian = math.ceil(self.purchased_license['Purchased_Licenses'
                                                   ].median())
        
        ax_license.text(0,0.5,"max: {0}\nmedian:{1}\nmin:{2}".
                        format(ymax,ymedian,ymin), 
                        bbox=dict(facecolor='gray'),
                        fontsize=8, transform=ax_license.transAxes)
        ax_license.set_yticks(ytick_list)
        ax_license.set_yticklabels(ytick_list, {'fontsize': 'xx-small'})
        ax_license.spines["right"].set_visible(False)
        ax_license.set_xlabel("days",{'fontsize': 'xx-small'})
        ax_license.grid(which='major', axis='both')
        ax_license.spines["top"].set_visible(False)
# =============================================================================
# drawing the unitary cost over time
# =============================================================================      
        ax_unitary_cost = drawing.add_subplot(grid[0:3,3:6])
        ax_unitary_cost.set_title('unitary cost -currency', 
                                  {'fontsize': 'x-small'})
        ax_unitary_cost.step(self.cost_t_s.index,
                             self.cost_t_s['unitary cost'],
                             where='post',
                             marker='o')
        
        ax_unitary_cost.set_xticks(xticks)
        ax_unitary_cost.set_xticklabels(xticks, {'fontsize': 'xx-small',
                                            'rotation': 45})
        ax_unitary_cost.grid(which='major', axis='both')
        ax_unitary_cost.set_xlabel("days",{'fontsize': 'xx-small'})
        ax_unitary_cost.spines["right"].set_visible(False)
        ax_unitary_cost.spines["top"].set_visible(False)
# =============================================================================
# mark the situation where the unitary cost has multiple value for the same
# point. This happens where the set of purchased licenses at a certain point
# in time is split into two clusters each of them with a different price.
# =============================================================================

# =============================================================================
# drawing the various license costs components (cumulated) over time
# =============================================================================
        
        ax_cumsum = drawing.add_subplot(grid[3:6,:-1])
        ax_cumsum.set_title('cumulated opex components - currency', 
                            {'fontsize': 'x-small'})
        
        x_values = self.cost_t_s.columns[2:]
        for i in range(len(pd_graph.index)):
            y_values = self.cost_t_s.iloc[i,2:].cumsum()
            ax_cumsum.plot(x_values, 
                           y_values,
                           label=self.cost_t_s.index[i], 
                           marker='d')
            
        ax_cumsum.set_xticks(xticks)    
        ax_cumsum.set_xticklabels(xticks, {'fontsize': 'xx-small',
                                            'rotation': 45})
        ax_cumsum.grid(which='major', axis='both')
        ax_cumsum.set_xlabel("days",{'fontsize': 'xx-small'})
        ax_cumsum.spines["right"].set_visible(False)
        ax_cumsum.spines["top"].set_visible(False)      
# =============================================================================
# drawing the license cost overall cumulated over time           
# =============================================================================
        pd_graph_cumsum_y=self.cost_t_s.cumsum()
        y_values = (pd_graph_cumsum_y.
                    iloc[len(pd_graph_cumsum_y.index)-1,2:].cumsum())
        ax_sum = drawing.add_subplot(grid[6:10,:-1])
        ax_sum.set_title('cumulated opex license currency', 
                         {'fontsize': 'x-small'})
        ax_sum.plot(x_values,
                    y_values,
                    marker='x')
        
        ax_sum.set_xticks(xticks)
        ax_sum.set_xticklabels(xticks, {'fontsize': 'xx-small',
                                            'rotation': 45})
        ax_sum.grid(which='major', axis='both')
        ax_sum.set_xlabel("days",{'fontsize': 'xx-small'})
        ax_sum.spines["right"].set_visible(False)
        ax_sum.spines["top"].set_visible(False)
        
        plt.savefig("./image")
        plt.show()
        
        return

# =============================================================================
# This is the start
# =============================================================================
def main ():
    """ main function definition"""   
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
# extract the purchase time sequence for licenses from the file defined in 
# the json configuration file
# =============================================================================
    default_license_file = "./licenses_time_sequence.txt"
    if 'License Profile' in initializator :
        license_t_s = initializator['License Profile']
    else:
        license_t_s =  default_license_file
    
    default_output_file = "./cost.csv"
    if 'Cost Result' in initializator:
        output_file = initializator['Cost Result']
    else:
        output_file = default_output_file

    
    with open(license_t_s) as license_file:
        license_purchase_history = pd.read_csv(license_file,';',header=0,
        names= ["time_in_day", "Purchased_Licenses"],index_col=0)
# =============================================================================
# extract the rule set for the maintenance calculation from the file defined 
# in json configuration file
# =============================================================================     
    default_rules_file = "./rules.txt"
    if 'Rules' in initializator :
        rules = initializator['Rules']
    else:
        rules =  default_rules_file
    
    with open(rules) as rules_file:
        rules_set = pd.read_csv(rules_file,';',header=0,
        names= ["cost", "<= t","t<","<=Licenses", "Licenses<"]) 
# =============================================================================
# initialize the opex object and evaluate the historical time series
# =============================================================================       
        
    opex = Opex(0) 
    opex.initialise(observed_time, output_file,
                    rules_set, license_purchase_history)
    opex.calculate()
    opex.plot(output_file)
    
    return

if __name__ == "__main__":
    main()

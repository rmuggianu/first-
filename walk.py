# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
import secrets 

class TreeMatrix:
    """Class that represents an binary Tree"""
    def __init__(self):
        return
        
    def setTree(self, steps, jumps, jumps_value):
        """initialize the Tree"""
        column_array = np.linspace(0,jumps*jumps_value, jumps+1,dtype=np.int64)
        self.tree = column_array.transpose()
        for i in range(steps-1):
            self.tree = np.column_stack((self.tree, column_array))
        self.first_step = 0
        self.steps = steps
        self.jumps = jumps
        return self.first_step

# =============================================================================
#     def __iter__(self):
#         return self
# 
#     def __next__(self):
#         if self.i < self.n:
#             i = self.i
#             self.i += 1
#             return i
#         else:
#             raise StopIteration() 
#             
# =============================================================================
    def getTreeNodeValue(self, step = 0, jump = 0):
        """retrieve the initial node of the tree"""
        return self.tree[jump, step]
    
    def getTreeNodeNextNumber(self) :
        """retrieve the number of next nodes"""   
        return (self.jumps+1)
    
# =============================================================================
# this methods takes the next Node in the NodeTree.It returns -1 if the node is
# a leaf it returns -1 (but it will modified throwing an exeption)
# =============================================================================
    def getNextStepTreeNode(self, step = 0):
        """Retrieve the nextnode Tree"""
        next_step = step
        if next_step < self.steps-1:
            next_step+=1
        else:
            next_step = -1
        return next_step

    def getNextTreeNodeList(self, step = 0, jump = 0):
        """Retrieve the nextnode Tree"""
        next_step = step
        if next_step < self.steps-1:
            next_step+=1
        else:
            next_step = -1
        return next_step
    
    def computePath(self, step = 0 , jump = 0, path=np.zeros(1),
                    path_matrix = np.zeros(1)):
    
        node_value = self.getTreeNodeValue(step,jump)
        path[step]=node_value
        next_step = self.getNextTreeNodeList(step, jump)
        if next_step != -1 :
            for i in range(self.getTreeNodeNextNumber()) :
                tmp_path_matrix = self.computePath(next_step, i,
                                   path, path_matrix)
                path_matrix = tmp_path_matrix
        else :
            tmp_path_matrix = np.vstack((path_matrix, path))
        return tmp_path_matrix


# output_file input is ignored for now
    def computeRandomPath(self, step = 0 , 
                          cumulated_license = 0,
                          jump = 0, path=np.zeros(1),
                          path_matrix = np.zeros(1), 
                          output_file = "./random_path.txt",
                          output_file_flag = 'False'):
    
        node_value = self.getTreeNodeValue(step,jump)
        path[step]=node_value
        next_step = self.getNextTreeNodeList(step, jump)
        if next_step != -1 :
            i = GetRandomicValue(self.getTreeNodeNextNumber())
            tmp_path_matrix = self.computeRandomPath(next_step, 
                                                     cumulated_license,
                                                     i,path, path_matrix,
                                                     output_file)
            path_matrix = tmp_path_matrix
        else :
            total=path.sum()
            path = (path/total)*cumulated_license          
            tmp_path_matrix = np.vstack((path_matrix, path))
        return tmp_path_matrix
# =============================================================================
# This creates an input time series where all the cumulated licenses in one 
# step.Output_file input is ignored for now
# =============================================================================
    def computeDiracPath(self, 
                         step = 0,
                         cumulated_license = 0,
                         path = np.zeros(1),
                         path_matrix = np.zeros(1), 
                         output_file = "./random_path.txt",
                         output_file_flag = 'False'):
        
        path = path*0
        for i in range(step):
            path[i] =  cumulated_license
            path_matrix = np.vstack((path_matrix,path)) 
            path[i] = 0
        tmp_path_matrix = path_matrix
        return tmp_path_matrix

# =============================================================================
# This creates an input time series distributing the dirac across all the steps
# Output_file input is ignored for now
# =============================================================================

    def computeDiracPathVector (self, 
                                step = 0,
                                cumulated_license = 0,
                                iteration_number = 1,
                                path = np.zeros(1),
                                path_matrix = np.zeros(1), 
                                output_file = "./random_path.txt",
                                output_file_flag = 'False'):
        

        counter = iteration_number/10        
        for i in range(iteration_number):
            if ((i%counter)== 0):
                print("processed items ", i)
#            path = path*0
#            for j in range(int(step/3)):
#                path[step- int(step/3) + j] =  cumulated_license*GetRandomicValue(step)
#            total=path.sum()
#            path = (path/total)*cumulated_license
#            path_matrix = np.vstack((path_matrix,path))
#            
#            path = path*0
#            for j in range(int(step/3)):
#                path[j] =  cumulated_license*GetRandomicValue(step)
#            total=path.sum()
#            path = (path/total)*cumulated_license
#            path_matrix = np.vstack((path_matrix,path)) 
            empty = np.zeros(0)
            path = path*0
            for j in range(step):
                path[j] =  GetRandomicValue(100)
            total=path.sum()
            path = (path/total)*cumulated_license
            path_matrix = np.vstack((path_matrix,path)) 
# np.roll(x,+1)
            duplicate = np.hstack((empty, path))
            for k in range(step-2):
                shift_path = path
                shift_path[step-k-1] = 0
                total = shift_path.sum()
                shift_path = (shift_path/total)*cumulated_license 
                path_matrix = np.vstack((path_matrix,shift_path))
            for k in range(step-2):
                shift_path = duplicate
                shift_path[k] = 0
                total = shift_path.sum()
                shift_path = (shift_path/total)*cumulated_license 
                path_matrix = np.vstack((path_matrix,shift_path))          
        tmp_path_matrix = path_matrix
        return tmp_path_matrix


def GetRandomicValue(top = 0):
    
     randomic_value = 0
     randomic_value = secrets.randbelow(top)
     return randomic_value


# =============================================================================
# This is for testing purposed
# =============================================================================
def main ():
    """ main function definition"""   
# =============================================================================
# Examples
# =============================================================================       
        
#    tree = TreeMatrix()
#    steps = 36
#    jumps = 150
#    tree.setTree(steps, jumps, 1)
#    computed_path = []
#    computed_path.append(tree.getTreeNodeValue())
#    next_step = tree.getNextStepTreeNode()
#    while next_step > 0:
#        computed_path.append(tree.getTreeNodeValue(step = next_step))
#        next_step = tree.getNextStepTreeNode(next_step)
    
    
    path = np.zeros(100)
#    all_path = np.zeros(steps)
    with open("./random_path_1.txt","w") as random_path:
        for i in range(1000):
            for j in range(100):
                path[j] =  GetRandomicValue(100)
            path.tofile(random_path, sep = ",")
            print("", file=random_path)
#    with open("./random_path.txt","w") as random_path:
#        for i in range(tree.getTreeNodeNextNumber()) :
#            all_path = tree.computeRandomPath(step=0, jump=i,path=path, 
#                                         path_matrix=all_path, 
#                                         output_file = random_path)

if __name__ == "__main__":
    main()

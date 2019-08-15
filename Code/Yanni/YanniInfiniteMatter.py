#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:52:03 2019

@author: YaniUdiani
"""

#!/usr/bin/env python

#------------------------------------------------------------------------------
# IMSRG_Magnus.py
#
# author:   Yani Udiani
# Adapted from Heiko's IMSRG
# date:     Jun, 15, 2019
# 
# tested with Python 3
# 
#This version is based off Exp.py
#It computes the self-energy (even though, I'm not going to use that here)
#It uses vstack in the PH transformation, and does a factor of 
#len(blk_nums_that_matter) dot products in the PH trans, so its a bit faster
#Also, it uses less instanciations of OP_Map()
#------------------------------------------------------------------------------

import numpy as np
from numpy import dot, reshape, transpose,block,vstack
#from scipy.integrate import odeint, ode
from numpy import linalg as LA # For Frobenius norm
from fractions import Fraction as Fr # For Bernoulli Numbers
#from sys import argv
import sys
import matplotlib.pyplot as plt # Plotting stuff
import copy
import time
#import numba
#from numba import jit
import itertools as tools #for infinite matter states

#-----------------------------------------------------------------------------------
# Ploting variables
#-----------------------------------------------------------------------------------
flow=[] #list of flow parameters
energy=[]#list of energies for plotting
eigs=[]
HH=[]
initializations = [] #used to store E_ref, and f_Hartree-fock
results = [] #used to store E_g.s, f,Gamma, Omega1B, Omega2B 
global_user_data =[]

Julie_Matrix = [] #Hamiltonian the way it was requested?
Julie_Flow = [] # values of s
#-----------------------------------------------------------------------------------
# Functions to compute Infinite Matter single particle states
#-----------------------------------------------------------------------------------
def powerset(iterable): #Function from itertools to find powerset
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return tools.chain.from_iterable(tools.combinations(s, r) for r in range(len(s)+1))

# The full_state generator below works by guessing and checking all microstates below the N_Max. 
# Below the N_Max is determined by <=sqrt(N_Max). Any state with a component of momenta greater than sqrt(N_Max) will be above N_Max.
# These fuctions only return "positive states": states with orbitals that have positive momenta
# Spin projections are added at the end of every microstate
# An example of a state is [0,0,0,0.5] 
def full_state_gen_3D(N_Max, degen): #3D "positive state" generator
    max_k_comp=np.floor(np.sqrt(N_Max))#define maximum wavenumber that won't exceed N_Max
    statey=[]
    if(degen == 2):#if you're in an isospin triplet (pure neutron matter for ex)
        for i in range(0,int(max_k_comp+1)):#int used ensure proper data type
            for j in range(0,int(max_k_comp+1)):
                for k in range(0,int(max_k_comp+1)):#+1 to ensure that the max k is included
                    if((i**2 + j**2 + k**2)<=N_Max):#check to make sure it is at or below the N_Max
                        statey.append([i, j, k,1/2])#including spin 1/2 particle. Sign_gen() will create the spin down counterpart
    
    if(degen == 4): #if you have both protons and neutrons
        for i in range(0,int(max_k_comp+1)):#int used ensure proper data type
            for j in range(0,int(max_k_comp+1)):
                for k in range(0,int(max_k_comp+1)):#+1 to ensure that the max k is included
                    if((i**2 + j**2 + k**2)<=N_Max):#check to make sure it is at or below the N_Max
                        statey.append([i, j, k,1/2,1/2])#including spin and isospin 1/2 particle. Sign_gen() will create the spin down counterparts
    return statey#returns list of acceptable states

#This function takes in "positive states" and returns the "negative states" corresponding to that state. 
#It also returns the postive state. For example, feeding in [0,1,0.5] will return [0,-1,0.5], [0,-1,-0.5],  [0,1,0.5]  and [0,1,-0.5]
#This is done to easily include the positive states and their negative equivalents in full_state variable below
def sign_gen(inputy):
    array=list(range(0,len(inputy)))
    results=list(powerset(array))
    all_sign_permutations=[]
    all_sign_permutations.append(inputy)
    for k in  range(1,len(results)):
        dummy_list=inputy[:]# Copies inputy to dummy_list
        if 0 not in [inputy[i] for i in results[k]]:#[inputy[i] for i in results[k]]=inputy[results[k]] in R speak
        ### input arrays with zeros will double count states, this condition takes care of that ####
            for j in results[k]:
                dummy_list[j]=-inputy[j]# flip signs
            all_sign_permutations.append(dummy_list)
            #print(dummy_list)
    return all_sign_permutations

#This function sorts states so that holes come before particles in Full_state. I prefer it this way. 
def sorter(temp_full_state, Fermi_lvl):
    holes=[]
    particles=[]
    for i in temp_full_state:
        if((i[0]**2+i[1]**2+i[2]**2)<=abs(Fermi_lvl)):
            holes.append(list(i))
        else:
            particles.append(list(i))
    return holes+particles

#The function below generates the particles and hole states that we're used to
#The function makes my infinite matter generator easily adaptable to my Magnus code
#This function assumes that the box is 3-D
def One_BD_States(Fermi_lvl, full_state): 
    particles=[]
# This loop will decrement from end of full_state to beginning, why? Cuz, the particles will be at the end!
    for i in range(len(full_state)-1, -1, -1): #range goes from len(full_state)-1 to 0 by units of -1
        if ((full_state[i][0]**2+full_state[i][1]**2+full_state[i][2]**2)> abs(Fermi_lvl)):#Note that this Fermi_lvl is the true Fermi_lvl
            #If it's a particle, append the index to the particle list
            #abs(Fermi_lvl) chosen to allow fermi lvl=-1 for easily testing some of my functions
            particles.append(i)
        else:#The moment it is not a particle, break. I'll manually tell it the indices that are holes.
            break
    holes=list(range(0, i+1))#holes are locations from 0 to i since i is below the fermi_lvl
    return holes, list(reversed(particles)) #reverse particles to ascending order because I looped backwards!

#-----------------------------------------------------------------------------------
# basis and index functions
#-----------------------------------------------------------------------------------
#Generates all possible pairs of single particle states and orders them such that the pairs that have the same
#sums of momenta are grouped together. It returns bas2B implicity organized in terms of blocks,
#it returns block_sizes which is an array of sizes of each block- generally useful
#it returns bas_block2B which organizes bas2B explicity in terms of blocks- will be useful for OP_Map class
#it returns Subset which is useful for constructing the Hamiltonian:
    
#For every pair (call it p) in Bas2B, the use of anti-symmetrized matrix elements allows me to calculate matrix elements of that pair
#with any pair in Bas2B, in addiiton to allowing me to simulatenously calculate matrix elements of pairs (call them g) in the permuation
#group of p with any pair in Bas2B. See lab notebook for detailed examples. Therefore, I need to loop over "unique" pairs in Bas2B
#when I'm calculating my 2-BD Hamiltonian. By "unique", I mean that I want one pair from each disjoint permuation group in Bas2B. This will save me on
#much needed compuation time. Also, I'm not going to consider terms like (r,r) in my matrix element so might as well select them out here as well. 
# The list subset2B is a filter of bas2B to including the unique pairs.

def construct_basis_2B(full_state,states):

    CM_Momenta=[]#list of all CM momenta
    pairs=[]
    block={}
    Bas2B=[]#gives all possible pairs grouped by CM momenta
    block_sizes=[]#number of pairs in a given block of Hamiltonian
    subset_sizes=[]#number of pairs in Subset
    
    for p in states:
        for q in states:
            CM_Momenta.append(full_state[p][0:3]+full_state[q][0:3])
            pairs.append((p,q))#store corresponding pairs for CM momenta
    
    for i, p in enumerate(CM_Momenta):
        if(str(p) not in block): #str(p) are keys corresponding to CM momenta blocks
            block[str(p)]=[pairs[i]]#Create a new block if it doesn't already exist
        else:
            if(pairs[i][0] <= pairs[i][1]): #store upper triangle first
                block[str(p)].append(pairs[i])#store pairs in a given block     
    
    num_blocks=len(block)#get number of blocks

    Subset=[]# List will contain lists of filterd pairs in a given block.
    #This list filters the pairs in each block. It acts as subset2B for each block    
    Bas_block2B=[[] for i in range(num_blocks)]# Bas2B with each block clearly demarcated
    block_num = 0
    
    for keys in block:
        #Remember that block[keys] is upper triangle including diag elements
        same_ij = [(i,j) for (i,j) in block[keys] if i==j]
        #same_ij stores (i,i) terms in block[keys] so that they'll be stuck at 
        #the very end of the block. That way, they don't interfere with my
        #mapping. I want to build my basis such that if I know the location of 
        #(i,j) for i!=j, then I know where (j,i) is. This is super useful!
        if same_ij: #if same_ij isn't empty
            blockness = [(i,j) for (i,j) in block[keys] if i!=j]#get terms without (i,i)
            Subset.append(blockness)#store upper triangle without diag (i,i)
            lower_T = [(i,j)[::-1] for (i,j) in blockness]#flip pairs
            block[keys] = blockness + lower_T + same_ij
        else:
            Subset.append(block[keys])#store upper triangle without diag (i,i)
            lower_T = [(i,j)[::-1] for (i,j) in block[keys]]#flip pairs
            block[keys] = block[keys] + lower_T
            
        for elements in block[keys]:
            Bas2B.append(elements) #get full Bas2B
            Bas_block2B[block_num].append(elements) #store pairs for blocks seperately
        block_sizes.append(len(block[keys]))#Number of pairs in each block
        block_num+= 1
                
    summ=0#test to ensure that I got the right number of pairs in Subset
    
    for k in range(len(Subset)):
        lenny=len(Subset[k])
        subset_sizes.append(lenny)
        summ+=lenny
        
    if((len(Bas2B)-len(Bas2B)**(.5))/2!=summ): #ensure that Subset represents upper triangular matrix of pairs
          print("Something is wrong with construct_basis_2B()")

    return Bas2B,Subset,block_sizes,subset_sizes,Bas_block2B

#Modifies bas_block2B to return all specific pairs that will be required by specific loops in
#commutator routine, and also eta construction
def pair_selector(bas_block2B, particles, holes):
    lenny = range(len(bas_block2B))
    Particle_Particle = [[] for i in lenny]
    Hole_Hole = [[] for i in lenny]
    Particle_Anything = [[] for i in lenny] #sperate P_H, and P_P pairs
    Hole_Anything = [[] for i in lenny]

    for block_num in lenny:
        for pairs in bas_block2B[block_num]:
            if(pairs[0] in particles):
                Particle_Anything[block_num].append(pairs)
                
                if(pairs[1] in particles and pairs[1] > pairs[0]):
                    Particle_Particle[block_num].append(pairs)
            else: #pairs[0] is in holes
                Hole_Anything[block_num].append(pairs)
                
                if(pairs[1] in holes and pairs[1] > pairs[0]):
                    Hole_Hole[block_num].append(pairs)      
                    
    return Particle_Particle, Hole_Hole, Particle_Anything, Hole_Anything 

#This function is used to speed up the 2B term in the 1B-2B comm
#For every block, and every term that is added to Output[(i,j), (k,l)]
#it determines which values of "a" will contribute
#left_connections corresponds to values of "a" where "a" is on the left most side
#of the pair : see the commutator eqs. Likewise for right connections 
#Note: I might change the labeling of "a" to something like "p" in the future
#to be consistent with chem notation.
def subset_selector(subset2B):
    
  left_connections=[{} for i in range(len(subset2B))]# contains sp states for looping for each blk
  right_connections=[{} for i in range(len(subset2B))]# contains sp states for looping for each blk
  
  for block_num, blocks in enumerate(subset2B):
    
    for (i,j) in blocks:
        
      if (i,j) not in left_connections[block_num].keys():#create lists in dict
        left_connections[block_num][(i,j)] = []
        right_connections[block_num][(i,j)] = []
        
      for (k,l) in blocks:
        if(j == l):
          left_connections[block_num][(i,j)].append(k)#corresponds to the a in states that is non-zero         
        if(i == k):
          right_connections[block_num][(i,j)].append(l)#corresponds to the a in states that is non-zero   
         
  return left_connections, right_connections
             
class OP_Map:#class maps to a given operator allowing calls to that operator's elements. These OPs are of the form [[]]
#This class is super useful because it allows me to keep a similar indexing structure of a single matrix even though I'm
#Using a list of block matrices for my operators
  
    #Initialize class vars to None. They'll be changed once the vars are known.
    obj = None #Each instance of OP_Map_B is a list operator, so obj will be 
    #routinely replaced locally by each instance: each instance will see its operator.
    idp = None #idp is shared across all instances of OP_Map_B, so it is a fixed
    #class var: it won't be changed locally or globally
    #idp_b = None #idp is shared across all instances of OP_Map_B, so it is a fixed
    #class var: it won't be changed locally or globally
    def __init__(self, tag):#use operator and idp to define self since those are universally
        self.tag = tag
        

    def __getitem__(self, pairs): # Will activate if you do a simple call on OP_Map
        first = OP_Map.idp[pairs[0]]
        second = OP_Map.idp[pairs[1]]
        if( first[0] != second[0] ):#if they aren't in the same block
            return 0
        else:            
            return self.obj[first[0]][first[1], second[1]]
        
    def __setitem__(self, pairs, val):# Activates when OP_Map()=val
        first = OP_Map.idp[pairs[0]]
        second = OP_Map.idp[pairs[1]]
        if( first[0] == second[0] ):#if they are in the same block
            self.obj[first[0]][first[1], second[1]] = val
            
    def __add__(self, pairs, val):# Activates when OP_Map()+=val
        first = OP_Map.idp[pairs[0]]
        second = OP_Map.idp[pairs[1]]
        if( first[0] == second[0] ):#if they are  in the same block
            self.obj[first[0]][first[1], second[1]] += val
        
    def __sub__(self, pairs, val):# Activates when OP_Map()-=val
        first = OP_Map.idp[pairs[0]]
        second = OP_Map.idp[pairs[1]]
        if( first[0] == second[0] ):#if they are in the same block
            self.obj[first[0]][first[1], second[1]] -= val   
   
class OP_Map_B:# Does same thing as OP_Map, but if the blk_num of both 
  #pairs is known, then it will search that block for their indices
  
  #Initialize class vars to None. They'll be changed once the vars are known.
  obj = None #Each instance of OP_Map_B is a list operator, so obj will be 
  #routinely replaced locally by each instance: each instance will see its operator.
  idp_b = None #idp_b is shared across all instances of OP_Map_B, so it is a fixed
  #class var: it won't be changed locally or globally
  
  def __init__(self, tag):
    self.tag = tag #dummy var: instance referenced through tag
    
  def __getitem__(self, pairs): # Will activate if you do a simple call on OP_Map_B
    #print(self.obj) 
    return self.obj[pairs[2]][OP_Map_B.idp_b[pairs[2]][pairs[0]], OP_Map_B.idp_b[pairs[2]][pairs[1]]]
        
  def __setitem__(self, pairs, val):# Activates when OP_Map_B()=val
    self.obj[pairs[2]][OP_Map_B.idp_b[pairs[2]][pairs[0]], OP_Map_B.idp_b[pairs[2]][pairs[1]]] = val
              
  def __add__(self, pairs, val):# Activates when OP_Map_B()+=val
    self.obj[pairs[2]][OP_Map_B.idp_b[pairs[2]][pairs[0]], OP_Map_B.idp_b[pairs[2]][pairs[1]]] += val
    
  def __sub__(self, pairs, val):# Activates when OP_Map_B()-=val
    self.obj[pairs[2]][OP_Map_B.idp_b[pairs[2]][pairs[0]], OP_Map_B.idp_b[pairs[2]][pairs[1]]] -= val
      
#Quick test to ensure that generic OP_Map does what it needs to do            
def OP_Map_Test(bas2B, block_sizes, bas_block2B, idp , H2B):
    
    H2B_Test = copy.deepcopy(H2B)
    block_num = block_sizes.index(min(block_sizes))#pick first smallest block
    block = bas_block2B[block_num]
    
    H2B_Test_Map = OP_Map("H2B_Test_Map")
    H2B_Test_Map.obj = H2B_Test
    
    #block=random.choice(bas_block2B)#pick random block (need to import random to work)
    #block_num=bas_block2B.index(block)
    for bra in block:
        for ket in block:           
            prev=H2B_Test[block_num][block.index(bra),block.index(ket)]
            
            ######## Can it call the appropriate matrix elements? ##########
            if(prev != H2B_Test_Map[[ bra, ket ]]):
                print("OP_Map fails at calling H2B(bra,ket) :",bra,ket,block_num)
                sys.exit()
            
            ######## Can it add/sub to matrix elements? ##########
            H2B_Test_Map[[ bra, ket ]]+=2 
            if(H2B_Test[block_num][block.index(bra),block.index(ket)] != prev + 2):
                print("OP_Map fails at adding to H2B(bra,ket) :",bra,ket,block_num)
                sys.exit()

                
            ######## Can it replace matrix elements? ##########
            H2B_Test_Map[[ bra, ket ]]=4            
            if(H2B_Test[block_num][block.index(bra),block.index(ket)] != 4):
                print("OP_Map fails at replacing H2B(bra,ket) :",bra,ket,block_num)
                sys.exit()

    return "Yaaay! OP_Map works as expected :-]"
    
#
# We use dictionaries for the reverse lookup of state indices
#
def construct_index_2B(bas2B):
  index = { }
  for i, state in enumerate(bas2B):
    index[state] = i

  return index

#The next function below does the same thing as idx2B, but stores the indices of pairs in a given block
#Figures out which block in which a given pair (p,q) lies. It also returns the index location of that 
#pair in bas_block2B: that is the row/column # of that given pair in Operator[block]
def special_index(bas_block2B):
  index = {} #dict containing pairs as keys to pair index -- not blocked
  index_blocked = {} #dict containing block_nums as keys to dicts wtih pairs as keys to pair index
  for block_num, bas in enumerate(bas_block2B):
    index_blocked[block_num] = {} 
    for pair in bas:
      ind = bas.index(pair)
      index[pair] = (block_num, ind)
      index_blocked[block_num][pair] = ind
      
  return index, index_blocked

#This function is for the 2BD terms in the commutators that use subset2B[block_num]
#This function reverses subset2B[block_num] in the enumeration, but still keeps the 
#initial indices for the pairs in subset2B[block_num]. This is because
#I want to do the loop over subset2B[block_num] backwards since I'll be deleting
#(i,j) from Hermitian- I don't want the indices of other pairs in Hermitian
#to change when I delete (i,j), so I start collecting (i,j) from the end of 
#subset2B[block_num]
def reverse_enumerate(subset2B_blk,length): 
    return zip(list(range(length-1, -1, -1)), reversed(subset2B_blk))
#-----------------------------------------------------------------------------------
# transform matrices to particle-hole representation
#-----------------------------------------------------------------------------------
def ph_transfrom_preparation(occphA_2B, block_sizes, bas_block2B,bs_len):
  blocks_that_matter=[] #list containing pairs in blocks that will be used in xph    
  blk_nums_that_matter=[occphA_2B[row][1] for row in range(len(occphA_2B))]#contain blk# for nonzero blks 
  y_ph_zeros = []
  for row in range(len(occphA_2B)): #loop over non-zero blocks of occupation matrix
      
    BN = blk_nums_that_matter[row]#block number of occphA_2B being considered
    blocks_that_matter+= bas_block2B[BN] #for computing PH trans on x
    y_ph_zeros.append([np.zeros((block_sizes[BN],block_sizes[i])) for i in bs_len])#initialize list of block matrices
      
  return blocks_that_matter,blk_nums_that_matter, y_ph_zeros  
      
def ph_transform_2B(x_map,y_map,occphA_2B, block_sizes,bas_block2B,
                    bas2B,bs_len,idp,subset2B,subset_size,
                    blocks_that_matter,blk_nums_that_matter,y_ph_zeros):

  List_Prod=[]
  y_ph_zeros = copy.deepcopy(y_ph_zeros)
  #blocks_that_matter=[] #list containing pairs in blocks that will be used in xph    
  #blk_nums_that_matter=[occphA_2B[row][1] for row in range(len(occphA_2B))]#contain blk# for nonzero blks 
  for row in range(len(occphA_2B)): #get PH transformation on y for each row that matters in Occ*ph(Y)
      
      BN = blk_nums_that_matter[row]#block number of occphA_2B being considered
      #blocks_that_matter+= bas_block2B[BN] #for computing PH trans on x
      
      #Instead of directly constructing the full matrix y_ph and then multiply that with the ful occphA_2B,
      #below, I'm constructing the blks in y_ph that will multiply with nonzero blks in occphA_2B
      #I then build the full matrix occphA_2B*y_ph using numpy.block()
      #y_ph=[np.zeros((block_sizes[BN],block_sizes[i])) for i in bs_len]#initialize list of block matrices
      y_ph = y_ph_zeros[row]
      #note that y_ph is not the same size as the full matrix version of y_ph.
      #Each block of y_ph is actually a row of the full matrix verision of y_ph
      
      for block_num in blk_nums_that_matter: #I'm looping over blk_nums_that_matter 
          #to avoid calculating terms in yph that will be zeroed out by zero occupation
          #matrices in occphA_2B. To see this, note that block_num needs to be in 
          #blk_nums_that_matter for y_ph[block_num] to be nonzero. Why?
          #Because the ph transformation preserves hermiticity.
          #But if BN wasn't in blk_nums_that_matter, then there wouldn't be a 
          #case where the first loop is over bas_block2B[block_num] and the 
          #second loop would be over bas_block2B[BN]. Then by hermiticty,
          #y_ph[block_num] would have to be zero.
          
          disp_2 = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)
          disp_1 = subset_size[BN]# = to how many pairs away (j,i) is from (i,j)
          for i1,(a,b) in enumerate(subset2B[BN]):
              for i2,(c,d) in enumerate(subset2B[block_num]):
                  y_ph[block_num][i1, i2] -=y_map[[(a,d),(c,b)]]
                  y_ph[block_num][i1 + disp_1, i2] -=y_map[[(b,d),(c,a)]]
                  y_ph[block_num][i1, i2 + disp_2] -=y_map[[(a,c),(d,b)]]
                  y_ph[block_num][i1 + disp_1, i2 + disp_2] -=y_map[[(b,c),(d,a)]]
      #note that pasting all y_ph[block_num] will correspond to partial row of
      #the full y_ph. I don't need the full row since the remaining matrix elems
      #will not contribute when I do Occ*ph(Y)
      
      #First combine block matrices (y_ph[block_num]) into 1 matrix of dimension
      #block_sizes[BN] * sum of all block_sizes, then multiply that matrix by
      # the applicable non-zero occupation matrix            
      y_ph = dot(occphA_2B[row][0],block(y_ph)) 
            

      List_Prod.append(y_ph) #append new row of concatenated block matrices
      
  Mat_Prod = vstack(List_Prod) #convert list into full matrix from rows of blocks
  #using vstack since I want to paste rows going down vertically
  x_ph = np.zeros((len(bas2B),len(blocks_that_matter))) 
  
  for i1, (a,b) in enumerate(bas2B):
    for i2, (c, d) in enumerate(blocks_that_matter):#only use columns that matter to construct x_ph
      x_ph[i1, i2] -= x_map[[(a,d),(c,b)]]
  return dot(x_ph, Mat_Prod)

        
def inverse_ph_transform_2B(Gamma_ph, block_sizes,idx2B,subset2B,bs_len,idp,subset_size):

  Gamma=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
  for block_num in bs_len:# go through blocks
      disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)
      for i1, (a, b) in enumerate(subset2B[block_num]):
          for i2, (c, d) in enumerate(subset2B[block_num]):
            Gamma[block_num][i1, i2]-=Gamma_ph[idx2B[(a,d)], idx2B[(c,b)]]
            Gamma[block_num][i1 + disp, i2]-=Gamma_ph[idx2B[(b,d)], idx2B[(c,a)]]
            Gamma[block_num][i1, i2 + disp]-=Gamma_ph[idx2B[(a,c)], idx2B[(d,b)]]
            Gamma[block_num][i1 + disp, i2 + disp]-=Gamma_ph[idx2B[(b,c)], idx2B[(d,a)]]
  
  return Gamma

def self_energy( Gamma,f, E, user_data ):
    
  #particles = user_data["particles"]
  #holes = user_data["holes"]
  states = user_data["states"]
  V = user_data["Gamma"]
  Particle_P = user_data["Particle_P"]
  Particle_A = user_data["Particle_A"]
  Hole_H = user_data["Hole_H"]
  Hole_A = user_data["Hole_A"]
  bs_len = user_data["bs_len"]
  
  Gamma_map = OP_Map_B("Gamma") #instance Gamma with attribute "Gamma"
  Gamma_map.obj = Gamma
  
  V_map = OP_Map_B("V") #instance intial Gamma with attribute "V"
  V_map.obj = V
  
  self_energy = np.zeros((len(states), len(states)))
  
  for p in states:
    for q in states:
      for block_num in bs_len:     
        for (i,r) in Hole_A[block_num]:
          for (j,s) in Hole_A[block_num]:
            if (i == j and r == p and s == q):
              for (a,b) in Particle_P[block_num]:
                  
                denom = E + f[i,i] - f[a,a] - f[b,b] 
                self_energy[p,q] = 0.5 * (Gamma_map[[(a,b), (i,p), block_num]] *
                           V_map[[(i,q), (a,b), block_num]])/denom    
 
        for (a,r) in Particle_A[block_num]:
          for (c,s) in Particle_A[block_num]:
            if (a == c and r == p and s == q):
              for (i,j) in Hole_H[block_num]:
                  
                denom = E + f[a,a] - f[i,i] - f[j,j] 
                self_energy[p,q] = 0.5 * (Gamma_map[[(i,j), (a,p), block_num]] *
                           V_map[[(a,q), (i,j), block_num]])/denom    
                             
                           
  return self_energy
#-----------------------------------------------------------------------------------
# commutator of matrices
#-----------------------------------------------------------------------------------
def commutator(a,b):
  return dot(a,b) - dot(b,a)

#-----------------------------------------------------------------------------------
# norms of off-diagonal Hamiltonian pieces and also, norm of 2-BD Lists
#-----------------------------------------------------------------------------------
def calc_fod_norm(f, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  
  norm = 0.0
  for a in particles:
    for i in holes:
      norm += f[a,i]**2 + f[i,a]**2

  return np.sqrt(norm)

def calc_Gammaod_norm(Gamma,user_data):
  
  Gamma_Map = OP_Map_B("Gamma")#create instance of OP_Map_B
  Gamma_Map.obj = Gamma #create class operator that it sees
  
  bs_len = user_data["bs_len"]
  #idp = user_data["idp"]
  #idp_b    = user_data["idp_b"]
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]


  norm = 0.0
  
  for block_num in bs_len:# go through blocks      
      if Hole_H[block_num]: #ensure that one of lists is not empty
          for bra in Particle_P[block_num] :#if Particle_P[block_num] is empty,
            for ket in Hole_H[block_num]:#no time will be wasted looping over Hole_H[blk#]
                norm += 4 * (Gamma_Map[[ bra , ket, block_num]]**2 
                        + Gamma_Map[[ ket , bra, block_num]]**2)

  return np.sqrt(norm)

def calc_full2B_norm(Gamma, subset2B,bs_len,idp_b):
    
  Gamma_Map = OP_Map_B("Gamma")#create instance of OP_Map_B
  Gamma_Map.obj = Gamma #create class operator that it sees
  
  norm = 0.0
  
  for block_num in bs_len:# go through blocks
      #find pairs in block      
      for bra in subset2B[block_num] :    
        for ket in subset2B[block_num]:
            norm += 4 * Gamma_Map[[ bra , ket, block_num]]**2  

  return np.sqrt(norm)

#-----------------------------------------------------------------------------------
# Functions used to do algebraic manipulations on lists of matrices (2BD operators)
#-----------------------------------------------------------------------------------
def List_dot(A, B, block_sizes,bs_len): #performs matrix multiplication on blocks of A and B with same index
    return [dot(A[i],B[i]) for i in bs_len]

def List_operation(A, B, block_sizes,sign,operation, bs_len): #performs simple operations on lists
    
    if(operation == "Transpose"): #will transpose A
      return [transpose(A[i]) for i in bs_len]
  
    if(operation == "List_Add"): #will do addition and subtraction of lists based on sign given
      return [A[i] + sign * B[i] for i in bs_len] 
  
    if(operation == "Scalar_Mul"): #will do multiplication of list element (Matrix A[i]) with scalar (B)
      return [A[i] * B for i in bs_len]
  
#    if(operation == "Scalar_Div"): #will do division of list element (Matrix A[i]) with scalar (B)  
#      return [A[i]/B for i in bs_len]
#    if(operation != "Transpose" or operation != "List_Add" or operation != "Scalar_Mul"):
#        print("You fed in an undefined operation into List_Operation")
#        sys.exit()

#-----------------------------------------------------------------------------------
# occupation number matrices
#-----------------------------------------------------------------------------------
def construct_occupation_1B(bas1B, holes, particles):
  dim = len(bas1B)
  occ = np.zeros(dim)

  for i in holes:
    occ[i] = 1.

  return occ

# diagonal matrix: n_a - n_b
def construct_occupationA_2B(block_sizes, bas_block2B, occ1B, bs_len):
  occ=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

  for block_num in bs_len:# go through blocks
    for i1, (i,j) in enumerate(bas_block2B[block_num]):
      occ[block_num][i1, i1] = occ1B[i] - occ1B[j]

  return occ


# diagonal matrix: 1 - n_a - n_b
def construct_occupationB_2B(block_sizes, bas_block2B, occ1B, bs_len):
  occ=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

  for block_num in bs_len:# go through blocks
    for i1, (i,j) in enumerate(bas_block2B[block_num]):
      occ[block_num][i1, i1] = 1. - occ1B[i] - occ1B[j]

  return occ

# diagonal matrix: n_a * n_b 
def construct_occupationC_2B(block_sizes, bas_block2B, occ1B, bs_len):
  occ=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

  for block_num in bs_len:# go through blocks
    for i1, (i,j) in enumerate(bas_block2B[block_num]):
      occ[block_num][i1, i1] = occ1B[i] * occ1B[j]

  return occ 

#-----------------------------------------------------------------------------------
# generators
#-----------------------------------------------------------------------------------
def eta_white(f, Gamma, user_data):
  #dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  block_sizes = user_data["block_sizes"]
  bs_len = user_data["bs_len"]
  #idp = user_data["idp"]
  #idp_b    = user_data["idp_b"]
  
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]
  
  Gamma_Map = OP_Map_B("Gamma")#create instance of OP_Map_B
  Gamma_Map.obj = Gamma #create class operator that it sees
  
  Gamma_Map_O = OP_Map("Gamma")#create instance of OP_Map
  Gamma_Map_O.obj = Gamma #create class operator that it sees
  
  # one-body part of the generator
  eta1B  = np.zeros_like(f)
  
  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i] + Gamma_Map_O[[(a,i),(a,i)]]
      val = f[a,i]/denom
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = [np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
  
  eta2B_map = OP_Map_B("eta2B")#create instance of OP_Map_B
  eta2B_map.obj = eta2B #create class operator that it sees
  
  for block_num in bs_len:# go through blocks
    
    if Hole_H[block_num]: 
      for (a,b) in Particle_P[block_num]:
        for (i,j) in Hole_H[block_num]:
          denom = ( 
          f[a,a] + f[b,b] - f[i,i] - f[j,j]  
          + Gamma_Map[[(a,b),(a,b),block_num]] 
          + Gamma_Map[[(i,j),(i,j),block_num]]
          - Gamma_Map_O[[(a,i),(a,i)]]
          - Gamma_Map_O[[(a,j),(a,j)]] 
          - Gamma_Map_O[[(b,i),(b,i)]]
          - Gamma_Map_O[[(b,j),(b,j)]]
          )
          val = Gamma_Map[[(a,b),(i,j),block_num]] / denom

          eta2B_map[[(a,b),(i,j),block_num]] = val
          eta2B_map[[(i,j),(a,b),block_num]] = -val
          
          eta2B_map[[(b,a),(i,j),block_num]] = -val
          eta2B_map[[(i,j),(b,a),block_num]] = val
          
          eta2B_map[[(a,b),(j,i),block_num]] = -val
          eta2B_map[[(j,i),(a,b),block_num]] = val
          
          eta2B_map[[(b,a),(j,i),block_num]] = val
          eta2B_map[[(j,i),(b,a),block_num]] = -val

  return eta1B, eta2B

def Bernoulli_generator(y): #returns list of Bernoulli numbers indexed by 0 to (y-1)
    def bernoulli2(): # Function taken online to calculate Bernoulli sequence
        A, m = [], 0
        while True:
            A.append(Fr(1, m+1))
            for j in range(m, 0, -1):
              A[j-1] = j*(A[j-1] - A[j])
            yield A[0] # (which is Bm)
            m += 1
    bn2 = [ix for ix in zip(range(y), bernoulli2())]
    bn3=[]
    for u in range(len(bn2)): # Convert messy format into easily accessible array 
        bn3.append(float(bn2[u][1]))
    return bn3

#insert remaning generators when you're ready to adapt them. 

#-----------------------------------------------------------------------------------
# Derivative routines
#-----------------------------------------------------------------------------------
#@jit  
def special_commutator(x, y, user_data,sign): # takes in either 1 or 2 body matrices to output 0,1, and 2 body commutators
#sign ensures that proper hermiticity is preserved in the particle hole transformation
#Since this commuatator is used for both the Hamiltonian calcuation with terms that look like ~[H,Omega] (hermitian with anti-hermitian)
#and Magnus expansion with terms that look like ~[Omega, eta] (anti-hermitian with anti-hermitian), the definition of xy in the 2B-2B
#commutator must be such that ~[H,Omega] returns a hermitian matrix; and ~[Omega, eta] returns an anti-hermitian matrix
    
    dim1B     = user_data["dim1B"]
    holes     = user_data["holes"]
    particles = user_data["particles"]
    
    idx2B     = user_data["idx2B"]
    idp       = user_data["idp"]
    #idp_b    = user_data["idp_b"]
    bas2B     = user_data["bas2B"]
    #basph2B   = user_data["basph2B"]
    #idxph2B   = user_data["idxph2B"]
    occB_2B   = user_data["occB_2B"]
    occC_2B   = user_data["occC_2B"]
    occphA_2B = user_data["occphA_2B"]
    
    subset_size = user_data["subset_sizes"]
    block_sizes = user_data["block_sizes"]
    bas_block2B = user_data["bas_block2B"]
    bs_len = user_data["bs_len"]
    
    Particle_P = user_data["Particle_P"]
    Hole_H = user_data["Hole_H"]
    Particle_A = user_data["Particle_A"]
    Hole_A = user_data["Hole_A"]
    subset2B = user_data["subset2B"]
    left_connections = user_data["left_connections"]
    right_connections = user_data["right_connections"]
    
    btm = user_data["blocks_that_matter"]
    bntm = user_data["blk_nums_that_matter"]
    ypz = user_data["y_ph_zeros"]
    
    #Initializations 
    Output_0B=0.0
    Output_1B=np.zeros((dim1B, dim1B))
    
    if(type(x) == np.ndarray and type(y) == np.ndarray):#1B-1B
        
        Output_2B = None #no 2B term
        Output_1B+=commutator(x,y)#regular commutator
        if(sign == 1): #omega_0B=-omega_0B
            for i in holes:# 0B Correction to reference E
                for j in particles:
                    Output_0B+=x[i,j]*y[j,i]-x[j,i]*y[i,j]
                
    if(type(x) == list and type(y) == np.ndarray):#2B-1B 
        
        x_map = OP_Map_B("x") #instance x with attribute "x"
        x_map.obj = x
        
        x_map_o = OP_Map("x") #instance x with attribute "x"
        x_map_o.obj = x
        
        Output_2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
        for block_num in bs_len:# go through blocks
            
            disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)
            for (a,j) in Hole_A[block_num]:
                for (b,i) in Particle_A[block_num]:
                    Output_1B[i,j]-= y[i,j]*( #changed this to a minus to mirror asymm of comm
                            x_map_o[[(a,i),(b,j)]]
                            -x_map[[(b,i),(a,j),block_num]])
                        
            Hermitian = copy.copy(subset2B[block_num])            
            for i1, (i,j) in reverse_enumerate(Hermitian,disp):
                for i2, (k,l) in enumerate(Hermitian):
                    mval = 0 #I hope this is stored in cache 
                    for a in left_connections[block_num][(i,j)]:
                        mval -= ( 
                                y[i,a] * x_map[[(a,j),(k,l),block_num]] 
                        )
                    for a in left_connections[block_num][(k,l)]:
                        mval += ( 
                                y[a,k] * x_map[[(i,j),(a,l),block_num]]
                        )

                    for a in right_connections[block_num][(i,j)]:
                        mval += ( 
                                y[j,a] * x_map[[(a,i),(k,l),block_num]] 
                        )
                    for a in right_connections[block_num][(k,l)]:
                        mval -= ( 
                                y[a,l] * x_map[[(i,j),(a,k),block_num]]
                        )                    
                        
                    Output_2B[block_num][i1, i2] = mval 
                    Output_2B[block_num][i1 + disp, i2]= -mval
                    Output_2B[block_num][i1, i2 + disp]= -mval
                    Output_2B[block_num][i1 + disp, i2 + disp]= mval
                    
                    Output_2B[block_num][i2, i1]= sign * mval
                    Output_2B[block_num][i2, i1 + disp]= -sign * mval
                    Output_2B[block_num][i2 + disp, i1]= -sign * mval
                    Output_2B[block_num][i2 + disp, i1 + disp]= sign * mval
                    
                del Hermitian[Hermitian.index((i,j))] 
                #delete (i,j) so it does show up in Hermitian since transpose is manually coded
                    
                            
    if(type(x) == np.ndarray and type(y) == list):#1B-2B 

        y_map = OP_Map_B("y") #instance y with attribute "y"
        y_map.obj = y
        
        y_map_o = OP_Map("y") #instance y with attribute "y"
        y_map_o.obj = y
        
        Output_2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
        for block_num in bs_len:# go through blocks
            
            disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)           
            for (a,j) in Hole_A[block_num]: 
                for (b,i) in Particle_A[block_num]:
                    Output_1B[i,j]+= x[i,j]*(
                            y_map_o[[(a,i),(b,j)]] #(a,i) not necessarily in same block as (b,i)
                            -y_map[[(b,i),(a,j),block_num]])
            
            Hermitian = copy.copy(subset2B[block_num])                
            for i1, (i,j) in reverse_enumerate(Hermitian,disp):
                for i2, (k,l) in enumerate(Hermitian):
                    mval = 0 #I hope this is stored in cache 
                    for a in left_connections[block_num][(i,j)]:
                        mval += ( 
                                x[i,a] * y_map[[(a,j),(k,l),block_num]] 
                        )
                    for a in left_connections[block_num][(k,l)]:
                        mval -= ( 
                                x[a,k] * y_map[[(i,j),(a,l),block_num]]
                        )

                    for a in right_connections[block_num][(i,j)]:
                        mval -= ( 
                                x[j,a] * y_map[[(a,i),(k,l),block_num]] 
                        )
                    for a in right_connections[block_num][(k,l)]:
                        mval += ( 
                                x[a,l] * y_map[[(i,j),(a,k),block_num]]
                        )                         
                    Output_2B[block_num][i1, i2] = mval
                    Output_2B[block_num][i1 + disp, i2]= -mval
                    Output_2B[block_num][i1, i2 + disp]= -mval
                    Output_2B[block_num][i1 + disp, i2 + disp]= mval
                    
                    Output_2B[block_num][i2, i1]= sign * mval
                    Output_2B[block_num][i2, i1 + disp]= -sign * mval
                    Output_2B[block_num][i2 + disp, i1]= -sign * mval
                    Output_2B[block_num][i2 + disp, i1 + disp]= sign * mval
                    
                del Hermitian[Hermitian.index((i,j))] 
                #delete (i,j) so it does show up in Hermitian since transpose is manually coded

    if(type(x) == list and type(y) == list):# 2B-2B
        
        xy= List_dot(x, List_dot(occB_2B, y, block_sizes, bs_len), block_sizes, bs_len)# xy has same blk structure
        xyz= List_dot(x, List_dot(occC_2B, y, block_sizes, bs_len), block_sizes, bs_len)#xyz has same blk structure
        
        #Output_2B= 0.5 * (xy + sign * transpose(xy))#sign reflects proper hermiticity
        
        Transpose = List_operation(xy, "NA", block_sizes,sign,"Transpose", bs_len)
        bracket = List_operation(xy, Transpose, block_sizes,sign,"List_Add", bs_len)
        Output_2B = List_operation(bracket, 0.5, block_sizes,sign,"Scalar_Mul", bs_len)
                
        x_map = OP_Map_B("x") #instance x with attribute "x"
        y_map = OP_Map_B("y") #instance y with attribute "y"
        xy_map = OP_Map_B("xy") #instance y with attribute "xy"
        xyz_map = OP_Map_B("xyz") #instance xyz with attribute "xyz"
        
        x_map.obj = x #set appropriate x object 
        y_map.obj = y #set appropriate y object 
        xy_map.obj = xy  #set appropriate xy object
        xyz_map.obj = xyz #set appropriate xyz object
        
        x_map_o = OP_Map("x") #instance x with attribute "x"
        y_map_o = OP_Map("y") #instance y with attribute "y"
        x_map_o.obj = x #set appropriate x object 
        y_map_o.obj = y #set appropriate y object
        
        xy_ph = ph_transform_2B(x_map_o,y_map_o,occphA_2B, block_sizes,
                                bas_block2B,bas2B,bs_len,idp,subset2B,
                                subset_size,btm,bntm,ypz)
        xyi = inverse_ph_transform_2B(xy_ph, block_sizes,idx2B,subset2B,bs_len,idp,subset_size)
        
        if(sign == 1):
            for block_num in bs_len:# go through blocks
                for (i,j) in Hole_H[block_num]:
                    for (k,l) in Particle_P[block_num]:
                        Output_0B += 2*(
                            x_map[[(i,j),(k,l),block_num]]
                            *y_map[[(k,l),(i,j),block_num]])
                                
        for block_num in bs_len:# go through blocks
            
            disp = subset_size[block_num]# = to how many pairs away (j,i) is from (i,j)  
            for (i,p) in Hole_A[block_num]:
                for (j,q) in Hole_A[block_num]:
                    if (j==i):
                        Output_1B[p,q]+=0.5*(xy_map[[(i,p), (i,q), block_num]] 
                        + sign * xy_map[[(i,q), (i,p), block_num]])
            
            Hermitian = copy.copy(subset2B[block_num])  
            for i1, (r,p) in reverse_enumerate(Hermitian,disp):
                for i2, (j,q) in enumerate(Hermitian):

                    if (j == r):
                        Output_1B[p,q]+=0.5*(xyz_map[[(r,p), (r,q), block_num]] 
                        + sign * xyz_map[[(r,q), (r,p), block_num]])
                        Output_1B[q,p] = sign * Output_1B[p,q]
                        
                    if (q == r):
                        Output_1B[p,j]+=0.5*(xyz_map[[(r,p), (r,j), block_num]] 
                        + sign * xyz_map[[(r,j), (r,p), block_num]])
                        Output_1B[j,p] = sign * Output_1B[p,j]
                        
                    if (j == p):
                        Output_1B[r,q]+=0.5*(xyz_map[[(p,r), (p,q),block_num]] 
                        + sign * xyz_map[[(p,q), (p,r),block_num]])
                        Output_1B[q,r] = sign * Output_1B[r,q]
                                               
                    if (q == p):
                        Output_1B[r,j]+=0.5*(xyz_map[[(p,r), (p,j),block_num]] 
                        + sign * xyz_map[[(p,j), (p,r), block_num]])
                        Output_1B[j,r] = sign * Output_1B[r,j]
                        
                    Output_2B[block_num][i1, i2]-=( 
                            xyi[block_num][i1, i2]
                            - xyi[block_num][i1 + disp, i2]
                            - xyi[block_num][i1, i2 + disp]
                            + xyi[block_num][i1 + disp, i2 + disp])
                    
                    mval = Output_2B[block_num][i1, i2] #I hope this gets put in cache
                    Output_2B[block_num][i1 + disp, i2]= -mval
                    Output_2B[block_num][i1, i2 + disp]= -mval
                    Output_2B[block_num][i1 + disp, i2 + disp]= mval
                    
                    Output_2B[block_num][i2, i1]= sign * mval
                    Output_2B[block_num][i2, i1 + disp]= -sign * mval
                    Output_2B[block_num][i2 + disp, i1]= -sign * mval
                    Output_2B[block_num][i2 + disp, i1 + disp]= sign * mval
                    
                del Hermitian[Hermitian.index((r,p))] 
                #delete (r,p) so it does show up in Hermitian since transpose is manually coded
                    
    return Output_0B, Output_1B, Output_2B   

def RHS_Cal(Omega_1B,Omega_2B,Eta_1B,Eta_2B, user_data):# Generates right hand side of flow equation to be fed to diffy solver to obtain new Omega
    bn2 = user_data["bn2"]
    block_sizes = user_data["block_sizes"]
    subset2B = user_data["subset2B"]
    bs_len = user_data["bs_len"]
    #idp = user_data["idp"]
    idp_b    = user_data["idp_b"]
    
    #RHS_0B=0. #Initialize RHS
    RHS_1B=np.zeros_like(Omega_1B)
    RHS_2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

    for n in range(len(bn2)):
        if n==0: 
            nth_2B=copy.deepcopy(Eta_2B) #Initial commutators defined to be Eta
            nth_1B=copy.copy(Eta_1B)

        else:
            C_1B_1B=special_commutator(Omega_1B,nth_1B, user_data, -1)#store one body-one body commutator
            C_1B_2B=special_commutator(Omega_1B,nth_2B, user_data, -1)#store one body-two body commutator
            C_2B_1B=special_commutator(Omega_2B,nth_1B, user_data, -1)#store two body-one body commutator
            C_2B_2B=special_commutator(Omega_2B,nth_2B, user_data, -1)#store two body-two body commutator
            #; -1 for anti-hermitian 2B piece
            # The next lines aggregate the current 0B, 1B, and 2B nth commutators from the Magnus expansion
      
            nth_1B=C_1B_1B[1]+C_1B_2B[1]+C_2B_1B[1]+C_2B_2B[1] #extract one body terms
            
            First_add=List_operation(C_2B_1B[2], C_2B_2B[2], block_sizes,+1,"List_Add", bs_len)
            nth_2B=List_operation(C_1B_2B[2], First_add, block_sizes,+1,"List_Add", bs_len)#extract two body terms
            
            
        # Next lines are from recursion relation from Baker–Campbell–Hausdorff formula modified including Bernoulli #s
        
        #shiftymatrix_0B=nth_0B/np.math.factorial(n)
        shiftymatrix_1B=nth_1B/np.math.factorial(n)
        
        shiftymatrix_2B=List_operation(nth_2B, 1/np.math.factorial(n), block_sizes,"NA","Scalar_Mul", bs_len)
        if ((LA.norm(shiftymatrix_1B)+ calc_full2B_norm(shiftymatrix_2B, subset2B,bs_len,idp_b)) < 1e-8 ):
            break        
        RHS_1B+=bn2[n]*shiftymatrix_1B
        
        First_mul = List_operation(shiftymatrix_2B, bn2[n], block_sizes,"NA","Scalar_Mul", bs_len)
        RHS_2B = List_operation(RHS_2B, First_mul, block_sizes,+1,"List_Add", bs_len)

    return 0.0, RHS_1B, RHS_2B

def Transformed_Ham(Omega_1B, Omega_2B, user_data):# Generates new Hamiltonian by performing BCH expansion
    bn2 = user_data["bn2"]
    block_sizes = user_data["block_sizes"]
    #subset2B = user_data["subset2B"]
    E=user_data["E"]
    f=user_data["f"]
    Gamma=user_data["Gamma"]
    bs_len = user_data["bs_len"]
    #idp = user_data["idp"]
    #idp_b    = user_data["idp_b"]
    
    H_0B=0. #Initialize Hamilitonian
    H_1B=np.zeros_like(f)
    H_2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices

    for n in range(len(bn2)):
        if n==0: 
            nth_2B=copy.deepcopy(Gamma) #Initial commutators defined to be H
            nth_1B=copy.copy(f)
            nth_0B=copy.copy(E)
        else:
            C_1B_1B=special_commutator(Omega_1B,nth_1B, user_data, +1)#store one body-one body commutator
            C_1B_2B=special_commutator(Omega_1B,nth_2B, user_data, +1)#store one body-two body commutator
            C_2B_1B=special_commutator(Omega_2B,nth_1B, user_data, +1)#store two body-one body commutator
            C_2B_2B=special_commutator(Omega_2B,nth_2B, user_data, +1)#store two body-two body commutator
            #; +1 for hermitian 2B piece
            # The next lines aggregate the current 0B, 1B, and 2B nth commutators from the Magnus expansion
            nth_0B=C_1B_1B[0]+C_2B_2B[0] #extract zero body terms       
            nth_1B=C_1B_1B[1]+C_1B_2B[1]+C_2B_1B[1]+C_2B_2B[1] #extract one body terms
            
            First_add=List_operation(C_2B_1B[2], C_2B_2B[2], block_sizes,+1,"List_Add", bs_len)
            nth_2B=List_operation(C_1B_2B[2], First_add, block_sizes,+1,"List_Add", bs_len)#extract two body terms
            
            
        # Next lines are from recursion relation from Baker–Campbell–Hausdorff formula modified including Bernoulli #s
        
        shiftymatrix_0B=nth_0B/np.math.factorial(n)
        shiftymatrix_1B=nth_1B/np.math.factorial(n)
        
        shiftymatrix_2B=List_operation(nth_2B, 1/np.math.factorial(n), block_sizes,"NA","Scalar_Mul", bs_len)
        #if ((LA.norm(shiftymatrix_1B)+ calc_full2B_norm(shiftymatrix_2B, subset2B,bs_len,idp_b)) < 1e-10 ):
        if (abs(shiftymatrix_0B) < 1e-4 ):#MeV
            #print(n)
            break        
        H_0B+=shiftymatrix_0B
        H_1B+=shiftymatrix_1B
        
        H_2B = List_operation(H_2B, shiftymatrix_2B, block_sizes,+1,"List_Add", bs_len)
        
    return H_0B, H_1B, H_2B
#------------------------------------------------------------------------------
# derivative wrapper
#------------------------------------------------------------------------------
def get_operator_from_y(y, dim1B, block_sizes, bs_len):
  
  #block_sizes = user_data["block_sizes"]
  two_body=[]
  # reshape the solution vector into 0B, 1B, 2B pieces
  ptr = 0
  zero_body = y[ptr]

  ptr += 1
  one_body = reshape(y[ptr:ptr+dim1B*dim1B], (dim1B, dim1B))
  ptr += dim1B*dim1B
  
  for i in bs_len:
      #dimBlk=block_sizes[i]*block_sizes[i]
      two_body.append(reshape(y[ptr : ptr+block_sizes[i]*block_sizes[i]], (block_sizes[i], block_sizes[i])))
      ptr += block_sizes[i]*block_sizes[i]

  return zero_body,one_body,two_body

def List_reshape(dOmega_2B,block_sizes, bs_len): #flatten 2B list into one array
    Output=[]
    for i in bs_len:
        Output.extend(reshape(dOmega_2B[i], -1))
        
    return Output

def derivative_wrapper(y, user_data, Transformed_Hamiltonian):

  dim1B = user_data["dim1B"]
  #dim2B = dim1B*dim1B
  calc_eta  = user_data["calc_eta"]# Function to calculate generator
  calc_rhs  = user_data["calc_rhs"]    #function to calculate RHS
  subset2B = user_data["subset2B"]
  block_sizes = user_data["block_sizes"]
  bs_len = user_data["bs_len"]
  #idp = user_data["idp"]
  idp_b    = user_data["idp_b"]
  
  # extract operator pieces from solution vector
  Omega_0B,Omega_1B, Omega_2B = get_operator_from_y(y, dim1B, block_sizes, bs_len)
  #Transform Hamiltonian using Omega
  #Transformed_Hamiltonian=Transformed_Ham(Omega_1B, Omega_2B, user_data)
  #print(Transformed_Hamiltonian[2][0])
  # calculate the generator
  Eta_1B, Eta_2B = calc_eta(Transformed_Hamiltonian[1], Transformed_Hamiltonian[2], user_data)
  
  # calculate the right-hand side
  dOmega_0B, dOmega_1B, dOmega_2B = calc_rhs(Omega_1B,Omega_2B,Eta_1B,Eta_2B, user_data)

  # convert derivatives into linear array
  dy   = np.append([dOmega_0B], np.append(reshape(dOmega_1B, -1), List_reshape(dOmega_2B,block_sizes, bs_len)))
  # share data
  #user_data["dE"] = dOmega_0B #storing dOmega_0B/ds in dE
  user_data["eta_norm"] = np.linalg.norm(Eta_1B, ord='fro') + calc_full2B_norm(Eta_2B, subset2B,bs_len,idp_b)
  
  return dy

#-----------------------------------------------------------------------------------
# Infinite Matter Hamiltonian
#-----------------------------------------------------------------------------------
def radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,alpha,L):#returns radial matrix elements (with no definite symm)
    #alpha corresponds to index of the arrays k_alpha and Combined_consts
    #p,q,r,s correspond to single particle states with wavevectors
    q = np.pi/L * ( full_state[p][0:3] - full_state[q][0:3] + full_state[s][0:3] - full_state[r][0:3])
    return Combined_consts[alpha] * np.exp(-np.dot(q,q)/(4 * k_alpha[alpha] ))

def spin_exch_element(p,q,r,s,full_state,ind):#returns spin matrix elements (with no definite symm)

    #ind=3 means that regular spin elements are computed; ind=4 means isospin elements are computed
    Element=0# intialize spin matrix element
    if(full_state[p][ind]==full_state[r][ind] and full_state[q][ind]==full_state[s][ind] ):
        Element+=0.5 + 2 * full_state[s][ind] * full_state[r][ind]
        
    if(full_state[p][ind]==full_state[r][ind]-1 and full_state[q][ind]==full_state[s][ind]+1 ):
        Element+=np.sqrt(3/4-full_state[r][ind] * (full_state[r][ind]-1)) * np.sqrt(3/4-full_state[s][ind] * (full_state[s][ind]+1))
    
    if(full_state[p][ind]==full_state[r][ind]+1 and full_state[q][ind]==full_state[s][ind]-1 ):
        Element+=np.sqrt(3/4-full_state[r][ind] * (full_state[r][ind]+1)) * np.sqrt(3/4-full_state[s][ind] * (full_state[s][ind]-1))
        
    return Element
    
def H2B_element(p,q,r,s,full_state,Combined_consts, k_alpha,degen,L):
    
    #if degen=1,then <t_pt_q|P^t|t_rt_s>=1 ->actually an isospin triplet state
    Element=0; A=0#These form the main pieces of the matrix element H2B[idx2B[(p,q)],idx2B[(r,s)]]
    spin = spin_exch_element(p,q,r,s,full_state,3)#precompute spin matrix elements
    if(degen == 4):#if I'm including isospin
        isospin= spin_exch_element(p,q,r,s,full_state,4)#precompute isospin matrix elements
        if(full_state[p][3]==full_state[r][3] and full_state[q][3]==full_state[s][3] 
        and full_state[p][4]==full_state[r][4] and full_state[q][4]==full_state[s][4]):
            A+=1
        A-= spin * isospin
        Element+=0.5 * A * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,0,L)#R term
        
        B=A; C=A
        if(full_state[p][4]==full_state[r][4] and full_state[q][4]==full_state[s][4]):
            B+= spin; C-= spin
        if(full_state[p][3]==full_state[r][3] and full_state[q][3]==full_state[s][3]):
            B-= isospin; C+= isospin
        Element+=0.25 * B * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,1,L)#T term
        Element+=0.25 * C * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,2,L)#S term
        
    if(degen == 2):# if I'm not including isospin
        if(full_state[p][3]==full_state[r][3] and full_state[q][3]==full_state[s][3]):
            A+=1
        A-= spin
        Element+=0.5 * A * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,0,L)#R term
        B = A + spin; C = A - spin
        if(full_state[p][3]==full_state[r][3] and full_state[q][3]==full_state[s][3]):
            B-= 1; C+= 1
        Element+=0.25 * B * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,1,L)#T term
        Element+=0.25 * C * radial_element(p,q,r,s,full_state,Combined_consts,k_alpha,2,L)#S term
        
    
    return Element #matrix element with no definite symmetry
    

    
def Inf_Matter_Ham(full_state, energy_const, Combined_consts, k_alpha, degen, L, user_data):
    
    states = user_data["states"]
    #bas2B = user_data["bas2B"]
    subset2B = user_data["subset2B"]
    block_sizes = user_data["block_sizes"]
    bas_block2B = user_data["bas_block2B"]
    bs_len = user_data["bs_len"]

    H1B=np.zeros((len(full_state), len(full_state)))
    for i in states:
        H1B[i,i] = energy_const * np.dot(full_state[i][0:3], full_state[i][0:3]) #0:3 doesn't include 3
        
    H2B=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
    for blocky in bs_len:# fill in all blocks
        #print(block)
        Hermitian=copy.copy(subset2B[blocky])# I plan to exploit H2B's hermiticity: 
        #Once a pair (p,q) has been considered in the bra, then it won't be considered 
        #in the ket via my loop. I will manually compute the transpose.
        for (p,q) in subset2B[blocky]:
            for (r,s) in Hermitian:
                #print([(p,q),(r,s)])
                
                #Get respective indices of pairs in each block
                block_loc_pq=bas_block2B[blocky].index((p,q))
                block_loc_rs=bas_block2B[blocky].index((r,s))
                block_loc_qp=bas_block2B[blocky].index((q,p))
                block_loc_sr=bas_block2B[blocky].index((s,r))
                
                #Compute anti-symmetrized matrix element <(p,q)| H2B |(r,s)>_AS
                if (np.all( full_state[p][0:3] + full_state[q][0:3] != full_state[r][0:3] + full_state[s][0:3] )):
                    print("CM Momentum is not being conserved!")
                #ensure that center of mass momentum is conserved for all matrix elements
                H2B[blocky][block_loc_pq,block_loc_rs] = (H2B_element(p,q,r,s,full_state,Combined_consts, k_alpha,degen,L)
                -H2B_element(p,q,s,r,full_state,Combined_consts, k_alpha,degen,L))#obtain anti-symm matrix element
                #Use properties of anti-symmetrized matrix elements
                H2B[blocky][block_loc_pq,block_loc_sr]= -H2B[blocky][block_loc_pq,block_loc_rs]
                H2B[blocky][block_loc_qp,block_loc_rs]= -H2B[blocky][block_loc_pq,block_loc_rs]
                H2B[blocky][block_loc_qp,block_loc_sr]=  H2B[blocky][block_loc_pq,block_loc_rs]
                
                #Use Hermiticity, also note that matrix elements are real
                H2B[blocky][block_loc_rs,block_loc_pq]= H2B[blocky][block_loc_pq,block_loc_rs]
                H2B[blocky][block_loc_sr,block_loc_pq]= H2B[blocky][block_loc_pq,block_loc_sr]
                H2B[blocky][block_loc_rs,block_loc_qp]= H2B[blocky][block_loc_qp,block_loc_rs]
                H2B[blocky][block_loc_sr,block_loc_qp]= H2B[blocky][block_loc_qp,block_loc_sr]
            del Hermitian[Hermitian.index((p,q))] #delete (p,q) so it does show up in Hermitian since transpose is manually coded
                    
    return H1B, H2B
 
#-----------------------------------------------------------------------------------
# Normal-ordered Infinite Matter Hamiltonian
#-----------------------------------------------------------------------------------
def normal_order(H1B, H2B, user_data):
  #This will rewrite the already second quantized Inf_matter Hamiltonian=H1B+H2B in terms of the Hartree Fock energy, the Fock operator, and H2B
  holes = user_data["holes"]
  states = user_data["states"]
  #idp    = user_data["idp"]
  H2B_Map_O = OP_Map("H2B")
  H2B_Map_O.obj = H2B
  #idp_b    = user_data["idp_b"]
  
  subset2B_holes=[(i, j) for i in holes for j in holes if i < j] #used to get 2B contribution to E in normal order routine
  subset2B_mixed=[(p, i) for p in states for i in holes if p != i] #used to get mean field contribution to 1B Hamiltonian
  # 0B part
  E = 0.0
  for i in holes:
    E += H1B[i,i]

  for p in subset2B_holes: #subset_holes has pairs (i,j) for i<j  --> allowed by exclusion principle. I add the corresponding (j,i) term using symmetry:
      #Since H2B[idx2B[(i,j)],idx2B[(i,j)]]=H2B[idx2B[(j,i)],idx2B[(j,i)]], I multiply  0.5*H2B[idx2B[p],idx2B[p]]
      #by a factor of 2. This way, I have a smaller loop range by a factor of 2. 
      E += H2B_Map_O[[p, p]]
      #print(H2B[idx2B[p],idx2B[p]])
      #print(p)

  # 1B part 
  f = copy.copy(H1B)
  for p in subset2B_mixed: #mean field contributions
    for q in subset2B_mixed:
      if(p[1] == q[1]):
        f[p[0], q[0]] += H2B_Map_O[[p, q]]  

  # 2B piece of new normal ordered Hamiltonian is H2B. No need for corrections to H2B

  return E, f, H2B

#def Cheap_Self_Energy(E, Gamma, f, particles, dim1B):
#  Sigma = np.zeros (dim1B, dim1B)
#-----------------------------------------------------------------------------------
# Perturbation theory
#-----------------------------------------------------------------------------------
def calc_mbpt2(f, Gamma, user_data):
  
  DE2 = 0.0
  bs_len = user_data["bs_len"]
  #idp    = user_data["idp"]
  #idp_b    = user_data["idp_b"]
  
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]

  Gamma_Map = OP_Map_B("Gamma") #instance of OP_Map with attribute "Gamma"
  Gamma_Map.obj = Gamma
  for block_num in bs_len:# go through blocks
      
    if Hole_H[block_num]:
      for (a,b) in Particle_P[block_num]:
        for (i,j) in Hole_H[block_num]:
           denom = f[i,i] + f[j,j] - f[a,a] - f[b,b]
           me    = Gamma_Map[[(a,b),(i,j),block_num]] 
           DE2  += me*me/denom

  return DE2

def calc_mbpt3(f, Gamma, user_data):

  particles = user_data["particles"]
  holes     = user_data["holes"]
  bs_len = user_data["bs_len"]
  #idp    = user_data["idp"]
  #idp_b    = user_data["idp_b"]
  
  Particle_P = user_data["Particle_P"]
  Hole_H = user_data["Hole_H"]
  
  Gamma_Map = OP_Map_B("Gamma") #instance of OP_Map with attribute "Gamma"
  Gamma_Map.obj = Gamma
  
  Gamma_Map_O = OP_Map("Gamma")#create instance of OP_Map
  Gamma_Map_O.obj = Gamma #create class operator that it sees
  
  DE3pp = 0.0
  DE3hh = 0.0
  DE3ph = 0.0
  
  for block_num in bs_len:# go through blocks
    
    for (i,j) in Hole_H[block_num]:    
      for (a,b) in Particle_P[block_num]:
        
        for (c,d) in Particle_P[block_num]:
          denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[i,i] + f[j,j] - f[c,c] - f[d,d])
          me    = (Gamma_Map[[(i,j),(a,b),block_num]]
          *Gamma_Map[[(a,b),(c,d),block_num]]
          *Gamma_Map[[(c,d),(i,j),block_num]])
          DE3pp += me/denom    
                         
        for (k,l) in Hole_H[block_num]:
          denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[l,l] - f[a,a] - f[b,b])
          me    = (Gamma_Map[[(a,b),(k,l),block_num]]
          *Gamma_Map[[(k,l),(i,j),block_num]]
          *Gamma_Map[[(i,j),(a,b),block_num]])
          DE3hh += me/denom  
          
        for k in holes:
          for c in particles:
            denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[j,j] - f[a,a] - f[c,c])
            me    = (Gamma_Map[[(i,j),(a,b),block_num]]#original expression
            *Gamma_Map_O[[(k,b),(i,c)]]
            *Gamma_Map_O[[(a,c),(k,j)]])
            DE3ph -= me/denom

            denom = (f[i,i] + f[j,j] - f[b,b] - f[a,a])*(f[k,k] + f[j,j] - f[b,b] - f[c,c]) 
            me    = (Gamma_Map[[(i,j),(b,a),block_num]]#flip (a,b)
            *Gamma_Map_O[[(k,a),(i,c)]]
            *Gamma_Map_O[[(b,c),(k,j)]])
            DE3ph -= me/denom

            denom = (f[j,j] + f[i,i] - f[a,a] - f[b,b])*(f[k,k] + f[i,i] - f[a,a] - f[c,c])
            me    = (Gamma_Map[[(j,i),(a,b),block_num]]#flip (i,j)
            *Gamma_Map_O[[(k,b),(j,c)]]
            *Gamma_Map_O[[(a,c),(k,i)]])
            DE3ph -= me/denom

            denom = (f[j,j] + f[i,i] - f[b,b] - f[a,a])*(f[k,k] + f[i,i] - f[b,b] - f[c,c])
            me    = (Gamma_Map[[(j,i),(b,a),block_num]]#flip (a,b) and (i,j)
            *Gamma_Map_O[[(k,a),(j,c)]]
            *Gamma_Map_O[[(b,c),(k,i)]])
            DE3ph -= me/denom
            
  return DE3pp+DE3hh+DE3ph

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

def main():
  ####Can change N_Max (determines # of particles),rho (determines size of box and Fermi lvl), and degen (type of matter) ######
  N_Max=0 #Define N_Max holes <= Nmax+1 determines holes and particles 
  rho=0.2 #Define density in fm^-3
  degen=2#If degen=4, then the matter is symmetric and isospin projections are considered. If degen=2, there is only
  #one type of particle, so isospin projections are not considered. Degen is only either 2 or 4. 
  
  temp_full_state=full_state_gen_3D(N_Max+1,degen)#this contains the "positive states" of the system
  #Notice that N_Max is shifted up by one to generate particle states. The true N_Max is still the one given above.
  #As the way it is coded now, all particle states above the given N_Max are used. Perhaps, it in the future, it may be best to 
  #Select the states above the given N_Max that I want to use for the calcuation. Just a thought....

  #############################
  temp_full_state=sorter(temp_full_state, N_Max)#order the states so that holes come before particle states in the list. I like it that way
  full_state=[]#This will contain all states in the system including the "negative states"
  for f in range(0,len(temp_full_state)):#loop over all positive states, and append their negative equivalents
      temp=sign_gen(temp_full_state[f])#get negative equivalents
      full_state.extend(temp)#extend used to append the elements of the list temp, not the entire list itself
  full_state=np.array(full_state)#covert it to an array for easy viewing
  #Note that full state corresponds to single particle states labled by mode #'s n, not wavenumbers k


  holes, particles=One_BD_States(N_Max, full_state)
  states = holes + particles
  A=len(holes) 
  L=(A/rho)**(1/3)
  Combined_consts=[(200/L**3)*(np.pi/1.487)**(3/2), -(178/L**3)*(np.pi/0.639)**(3/2), -(91.85/L**3)*(np.pi/0.465)**(3/2)]
  #Combined constants for matrix element. 
  k_alpha=[1.487,0.639,0.465] #R,T,S
  energy_const=(197.3269)**2/(2*939.565)*(2*np.pi/L)**2 #hbar^2/2m in MeV fm^2
  
  # setup shared data
  dim1B = len(holes)+len(particles)
  #dim2B = dim1B * dim1B
  # basis definitions
  bas1B     = range(dim1B)
  bas2B,subset2B,block_sizes,subset_sizes,bas_block2B = construct_basis_2B(full_state,states)
  Particle_P,Hole_H,Particle_A,Hole_A = pair_selector(bas_block2B, particles, holes)
  #print(Hole_H)
  left_connections,right_connections = subset_selector(subset2B)
  bs_len = range(len(block_sizes)) #:O
  #basph2B   = construct_basis_ph2B(holes, particles)
  idx2B     = construct_index_2B(bas2B)
  idp, idp_b = special_index(bas_block2B)
  
  #Initialize OP_Map with fixed variables idp, idp_b
  OP_Map_B.idp_b = idp_b
  OP_Map.idp_b = idp_b
  OP_Map.idp = idp
  #idxph2B   = construct_index_2B(bas2B)

  # occupation number matrices
  occ1B     = construct_occupation_1B(bas1B, holes, particles)
  occA_2B   = construct_occupationA_2B(block_sizes, bas_block2B, occ1B, bs_len)
  occB_2B   = construct_occupationB_2B(block_sizes, bas_block2B, occ1B, bs_len)
  occC_2B   = construct_occupationC_2B(block_sizes, bas_block2B, occ1B, bs_len)

  occphA_2B = construct_occupationA_2B(block_sizes, bas_block2B, occ1B, bs_len)
  occphA_2B=[[occphA_2B[i],i] for i in bs_len 
  if np.all( occphA_2B[i] == np.zeros( np.shape(occphA_2B[i]) )) !=True]
  blocks_that_matter,blk_nums_that_matter, y_ph_zeros =(
          ph_transfrom_preparation(occphA_2B, block_sizes, bas_block2B,bs_len))
  #Now occphA_2B is composed of blocks with non-zero occupation # matrices, and
  #their corresponding block numbers.
  
  bn2=Bernoulli_generator(16)# Go up to 16 terms in expansion
  bn2[1]=-0.5#follow 2nd convention for Bernouli numbers   
  
  # store shared data in a dictionary, so we can avoid passing the basis
  # lookups etc. as separate parameters all the time
  user_data  = {
    "dim1B":      dim1B, 
    "holes":      holes, 
    "particles":  particles,
    "bas1B":      bas1B,
    "bas2B":      bas2B,
    "subset2B":   subset2B,
    "bas_block2B": bas_block2B,
    "subset_sizes": subset_sizes,
    "block_sizes": block_sizes,
    "Particle_P": Particle_P,
    "Hole_H":     Hole_H,
    "Particle_A": Particle_A,
    "Hole_A":     Hole_A,
    "left_connections": left_connections,
    "right_connections": right_connections,
    "bs_len":     bs_len,
    #"basph2B":    basph2B,
    "idx2B":      idx2B,
    "idp":        idp,
    "idp_b":      idp_b,
    #"idxph2B":    idxph2B,
    "occ1B":      occ1B,
    "occA_2B":    occA_2B,
    "occB_2B":    occB_2B,
    "occC_2B":    occC_2B,
    "occphA_2B":  occphA_2B,
    "blocks_that_matter": blocks_that_matter,
    "blk_nums_that_matter": blk_nums_that_matter,
    "y_ph_zeros": y_ph_zeros,
    "bn2": bn2,
    # variables for sharing data between ODE solver
    "eta_norm":   0.0,                #Initial norm of eta
    "dE":         0.0,                # Initial dOmega/ds
    "calc_eta":   eta_white,          # specify the generator (function object)
    "calc_rhs":   RHS_Cal,         # specify the right-hand side and truncation
    "special_commutator": special_commutator, #commutator used to evaluate 1-2 body commutators
    "states": states,
    #"OP_Map": OP_Map
  }

  global_user_data.append(user_data) # for use in other scripts
  # set up initial Hamiltonian
  H1B,H2B=Inf_Matter_Ham(full_state, energy_const, Combined_consts, k_alpha, degen, L, user_data)
  print(OP_Map_Test(bas2B, block_sizes, bas_block2B, idp, H2B))#ensure that OP_Map is doing what it needs

  E, f, Gamma = normal_order(H1B, H2B, user_data)#Normal ordered Hamiltonian with Hartree Fock energy
  HH.append(H2B)
  initializations.append([E,f, Gamma])
  
  #eigs.extend(np.linalg.eig(Gamma)[0])#store eigenvalues of H2B. This is interesting
  
  # Append Initial Hamiltonian to dictionary since that won't be changing in the Magnus expansion
  user_data["E"]= E
  user_data["f"]= f
  user_data["Gamma"]= Gamma
  
  #Initializations
  Initial_Omega0=0.
  Initial_Omega1=np.zeros((dim1B,dim1B))
  Initial_Omega2=[np.zeros((block_sizes[i],block_sizes[i])) for i in bs_len]#initialize list of block matrices
  #Transformed_Hamiltonian = Transformed_Ham(Initial_Omega1, Initial_Omega2, user_data)
  Transformed_Hamiltonian = E, f, Gamma #initial transformation preserves H
  
  # reshape generator into a linear array (initial ODE vector)
  y0   = np.append([Initial_Omega0], np.append(reshape(Initial_Omega1, -1), List_reshape(Initial_Omega2,block_sizes, bs_len)))
  Omega_F= y0# flattened Omega
  
  # integrate flow equations 
  sinitial=0
  sfinal = 50  
  ds = 0.01
  num_points = (sfinal-sinitial)/ds +1
  flow_pars = np.linspace(sinitial,sfinal,int(num_points))
  print( "Reference Energy (MeV):", round(E,4))
  #print(eta_white(f,Gamma, user_data)[0])
  
  print ("%-14s   %-11s   %-14s   %-14s   %-14s  %-14s   %-14s   %-14s"%(
    "s", "E" , "DE(2)", "DE(3)", "E+DE", 
    "||eta||", "||fod||", "||Gammaod||"))
  # print "-----------------------------------------------------------------------------------------------------------------"
  print ("-" * 130)
  start_time = time.time()
  
  for flow_p in flow_pars:
    ys = ds * derivative_wrapper(Omega_F, user_data, Transformed_Hamiltonian) + Omega_F

    Omega_0B,Omega_1B, Omega_2B = get_operator_from_y(ys, dim1B, block_sizes,bs_len)#get new Omegas
    
    Transformed_Hamiltonian = Transformed_Ham(Omega_1B, Omega_2B, user_data)#get  Hamiltonian to use MBPT, and print out values
    E, f , Gamma = Transformed_Hamiltonian #get  Hamiltonian to use MBPT, and print out values
    
    ###### Stuff For Julie ######
    temp_H = [E]
    temp_H.extend(reshape(f,-1))
    temp_H.extend(List_reshape(Gamma,block_sizes,bs_len))
    temp
    Julie_Matrix.append(temp_H)
    Julie_Flow.append(flow_p)
    
    #print("Omega",LA.norm(Omega_1B))
    #print("f",LA.norm(f-np.diag(np.diag(f))))
    energy.append(E)#append energy for plotting
    flow.append(flow_p)#append flow parameter for plotting


    DE2 = calc_mbpt2(f, Gamma, user_data)
    DE3 = calc_mbpt3(f, Gamma, user_data)

    norm_fod     = calc_fod_norm(f, user_data)
    norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)

    #print ("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
        #flow_p, E , DE2, DE3, E+DE2+DE3, user_data["eta_norm"], norm_fod, norm_Gammaod))
    if (abs(DE2/E) < 10e-8): 
        print("Time Taken to Run--- %s Minutes ---" % (round((time.time() - start_time)/60,4)))
        print( "Correlation Energy (MeV):", round(E-user_data["E"],4))
        results.append([E,f,Gamma,Omega_1B,Omega_2B])
        break
    Omega_F=np.append([Omega_0B], np.append(reshape(Omega_1B, -1), List_reshape(Omega_2B,block_sizes, bs_len)))
  np.save('yanni_matrix_1e-2', Julie_Matrix)
  np.save('yanni_flow_1e-2', Julie_Flow)
#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
main()    
#if __name__ == "__main__": 
#  main()
#plt.plot(flow, energy)

# Configuration Interaction code for exact diagonalization.

# Author: Unknown
# Modified by: Jaocb Davison
# Date: 07/31/19

import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
#%matplotlib inline #only if using in jupyter

up_states = [[1,1,0,0],
             [1,0,1,0],
             [1,0,0,1],
             [0,1,1,0],
             [0,1,0,1],
             [0,0,1,1]]
dn_states = up_states


states = [] # [[phase, up_vector, dn_vector], ...]
for i in range(0, len(up_states)):
    for j in range(0, len(dn_states)):
            states.append([1, copy.deepcopy(up_states[i]), copy.deepcopy(dn_states[j])])

def number_of_pairs(state): # output number of pairs
    pairs = 0
    state_up = state[1]
    state_down = state[-1]
    for i in range(0,len(state_up)):
        if state_up[i] == state_down[i] == 1:
            pairs += 1
    return pairs

#sort by number of pairs
states.sort(key=number_of_pairs, reverse=True)
# for i in range(0,len(states)):
#     print("%i:  %s "%(i, states[i]))


def inner_product(state_1, state_2):
    if (state_1[1] == state_2[1]) and (state_1[2] == state_2[2]):
        IP = 1;
    else:
        IP = 0;
    return state_1[0]*state_2[0]*IP # phases * inner product


def creator(spin, i, state): # spin: 1=up, -1=down | i=index to operate on
    vec = copy.deepcopy(state)
    n = 0 # number of occupied states left of i
    for bit in vec[spin][0:i]:
        if bit == 1: n += 1

    if vec[spin][i] == 0: # create
        vec[spin][i] = 1
        vec[0] *= (-1)**n # phase
        return vec
    else:
        vec[0] = 0
        return vec

def annihilator(spin, i, state): # spin: 1=up, -1=down | i=index to operate on
    vec = copy.deepcopy(state)
    n = 0 # number of occupied states left of i
    for bit in vec[spin][0:i]:
        if bit == 1: n += 1

    if vec[spin][i] == 1: # annihilate
        vec[spin][i] = 0
        vec[0] *= (-1)**n # phase
        return vec
    else:
        vec[0] = 0
        return vec



def delta_term(phi, phip, p, d):
    result = 0
    temp = annihilator(1, p, phip)
    temp = creator(1, p, temp)
    result += (p)*d*inner_product(phi,temp)

    temp = annihilator(-1, p, phip)
    temp = creator(-1, p, temp)
    result += (p)*d*inner_product(phi,temp)
    #print("----D-result: ", result)
    return result


def g_term(phi, phip, p, q, g):
    result = 0
    temp = annihilator(1, q, phip)
    temp = annihilator(-1, q, temp)
    temp = creator(-1, p, temp)
    temp = creator(1, p, temp)
    result += (-g/2)*inner_product(phi,temp)
    #print("----G-result: ", result)
    return result


def f_term(phi, phip, p, q, pp, f):
    if p == pp:
        return 0
    else:
        result = 0
        temp = annihilator(1, q, phip)
        temp = annihilator(-1, q, temp)
        temp = creator(-1, pp, temp)
        temp = creator(1, p, temp)
        # if temp[0]!=0:
        #     print("p=%i, pp=%i, q=%i"%(p,pp,q))
        #     print(temp)
        #     print(phi)
        #     print(inner_product(phi,temp))
        result += (f/2)*inner_product(phi,temp)

        temp = annihilator(1, pp, phip)
        temp = annihilator(-1, p, temp)
        temp = creator(-1, q, temp)
        temp = creator(1, q, temp)
        result += (f/2)*inner_product(phi,temp)
        # print("----F-result: ", result)
        return result




def matrix(states, d, g, f): # states is a list of form [[phase, up, down], [] ... []]
    H = np.zeros((len(states),len(states)))
    row, col = -1, -1 # increases to 0 at first run through
    length = len(states[0][1])

    for phi in states:
        col = -1 # reset col number to be increased to 0
        row += 1
        for phip in states:
            col += 1
            for p in range(0,length):
                H[row, col] += delta_term(phi, phip, p, d)
                for q in range(0,length):
                    H[row,col] += g_term(phi, phip, p, q, g)
                    for pp in range(0,length):
                        #print("[{},{}] p:{}, q:{}, pp:{}".format(row,col,p,q,pp))
                        #print(phi, phip)
                        H[row,col] += f_term(phi, phip, p, q, pp, f)

    return H

def exact_diagonalization(d, g, pb):
    print("""\nRunning full CI diagonalization for
    d              = {:2.4f}
    g              = {:2.4f}
    pb             = {:2.4f}""".format(d, g, pb))
    hme = matrix(states, d, g, pb)

    w,v = np.linalg.eig(hme)
    w_sort = w[np.argsort(w)]
    print("--- Ground state energy: {:2.4f}\n".format(w_sort[0]))
    return w_sort[0] # ground state energy
#
# hme = matrix(states, 1.0, 0.5, 0.1) # delta, g, f
#
# w,v = np.linalg.eig(hme)
# print(w[np.argsort(w)])
# np.set_printoptions(threshold=sys.maxsize, linewidth=300) # show full array
# #print(hme)
#
#
# # large map of hme
# fig = plt.figure(figsize=(8,8))
# # plt.imshow(hme, cmap='summer',
# #                 interpolation='nearest',
# #                 vmin=np.amin(hme),
# #                 vmax=np.amax(hme))
# # plt.imshow(hme, cmap='RdBu_r',
# #                 interpolation='nearest',
# #                 vmin=min(np.amin(hme), -np.amax(hme)),
# #                 vmax=max(np.amax(hme), np.abs(np.amin(hme))))
# plt.imshow(hme, cmap='RdBu_r',
#                 interpolation='nearest',
#                 vmin=np.amin(hme),
#                 vmax=np.amax(hme))
# plt.tick_params(axis='y', which='both', labelleft=False, labelright=False)
# plt.tick_params(axis='x', which='both', labelbottom=False, labeltop=False)
#
# # # Loop over data dimensions and create text annotations.
# # for i in range(len(hme)):
# #     for j in range(len(np.array(hme)[0])):
# #         if np.abs(hme[i,j]) > 1.0e-5: # only if above threshold
# #             plt.text(j, i, round(hme[i, j], 2), ha="center", va="center", color="black")
#
# plt.show()

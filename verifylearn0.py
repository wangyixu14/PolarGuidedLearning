import numpy as np
import os
import wasserstein as was
import torch
import time
import sys

NUM = sys.argv[1]
ID = '_' + NUM

def get_params():
    hidden_size = 2
    all_param = []
    init_param = np.random.rand(4*hidden_size+1)
    init_param = init_param.tolist()

    all_param += init_param
    all_param += [0, 1]
    all_param = [2, 1, 1, hidden_size] + all_param

    all_param[6] = 0
    all_param[9] = 0
    all_param[12] = 0
    # all_param[22] = 0

    np.savetxt('systems_with_networks/reachnn_oscillator_benchmark/nn_test_relu_tanh'+ID, np.array(all_param))
    return np.array(all_param)

class set:
    def __init__(self, x):
        assert x[2] > x[0]
        assert x[3] > x[1]
        self.x_inf = x[0]
        self.y_inf = x[1]
        self.x_sup = x[2]
        self.y_sup = x[3]
        self.area = (self.x_sup - self.x_inf)*(self.y_sup - self.y_inf)

def geometry(myreachset, mytargetset, printC = False):
    # the width and height of 2 rectangles
    reachset = set(myreachset)
    targetset = set(mytargetset)
    r_width = reachset.x_sup - reachset.x_inf
    r_height = reachset.y_sup - reachset.y_inf
    g_width = targetset.x_sup - targetset.x_inf 
    g_height = targetset.y_sup - targetset.y_inf

    # the center position of two rectangles
    r_x = (reachset.x_inf + reachset.x_sup) / 2
    r_y = (reachset.y_inf + reachset.y_sup) / 2
    g_x = (targetset.x_inf + targetset.x_sup) / 2
    g_y = (targetset.y_inf + targetset.y_sup) / 2
    
    inter = False
    if intersect(myreachset, mytargetset):
        # to compute the intersection area
        overlapX = r_width + g_width - (max(reachset.x_sup, targetset.x_sup)-min(reachset.x_inf, targetset.x_inf))
        overlapY = r_height + g_height - (max(reachset.y_sup, targetset.y_sup)-min(reachset.y_inf, targetset.y_inf))
        assert overlapX >= 0
        assert overlapY >= 0
        area = overlapX*overlapY
        if printC:
            print("Ohoo, intersected and the area is: ", area)
        inter = True
        return -area
    else:
        # to compute the minimal distance between two sets
        # the distance of two centers on x, y
        Dx = abs(r_x - g_x)
        Dy = abs(r_y - g_y)

        if Dx < (r_width+g_width) / 2 and Dy >= (r_height+g_height) / 2:
            min_dist = Dy - (r_height+g_height) / 2
        elif Dx >= (r_width+g_width) / 2 and Dy < (r_height+g_height) / 2:
            min_dist = Dx - (r_width+g_width) / 2
        elif Dx >= (r_width+g_width) / 2 and Dy >= (r_height+g_height) / 2:
            delta_x = Dx - (r_width+g_width) / 2
            delta_y = Dy - (r_height+g_height) / 2
            min_dist = np.sqrt(delta_x**2+delta_y**2)
        if printC:
            print("Not intersect and min distance is: ", min_dist)
        return min_dist 

def W_distance(reachset, goalset, printC):

    N = 100
    x0 = np.random.uniform(reachset[0], reachset[2], size=(N,))
    x1 = np.random.uniform(reachset[1], reachset[3], size=(N,))
    x = np.vstack((x0, x1))

    y0 = np.random.uniform(goalset[0], goalset[2], size=(N,))
    y1 = np.random.uniform(goalset[1], goalset[3], size=(N,))
    y = np.vstack((y0, y1))

    X = torch.FloatTensor(x.T)
    Y = torch.FloatTensor(y.T)
    # OS
    epsilon = 0.01
    niter = 100

    # ACC
    epsilon = 100
    niter = 200
    l1 = was.sinkhorn_loss(X,Y,epsilon,N,niter)
    # l2 = was.sinkhorn_normalized(X,Y,epsilon,N,niter)
    if intersect(reachset, goalset): 
        if printC:
            print('ohh, intersect and W distance is: ', l1.item())
    else:
        if printC:
            print('not intersect and W distance is: ', l1.item())   
    return l1.item()    

def intersect(reachset, set2):
    assert reachset.shape == (4, )
    if reachset[2] <= set2[0] or reachset[0] >= set2[2] or reachset[3] <= set2[1] or reachset[1] >= set2[3]:
        return False    
    else:
        return True

def gradient(it, control_param, goalset, unsafeset):

    goalgrad = np.zeros(len(control_param), )
    safetygrad = np.zeros(len(control_param), )
    delta = np.random.uniform(low=-0.05, high=0.05, size=(control_param.size))
    index_list = [4, 5, 7, 8, 10, 11]

    for index in index_list:
        if index <= 5:
            printC = True
        else:
            printC = False 
        pert = np.zeros(len(control_param), )
        pert[index] += delta[index]
        np.savetxt('systems_with_networks/reachnn_oscillator_benchmark/nn_test_relu_tanh'+ID, control_param+pert)
        os.system('./benchmark_oscillator_relu_tanh 0.01 20 4 6 0 '+str(NUM))
        # assert False
        reachset = np.load('StepReach0'+ID+'.npy')
        reachset = np.reshape(reachset, (-1, 4))

        goalreached = intersect(reachset[-1, :], goalset)
        unsafe = False
        for i in range(len(reachset)):
            unsafe = unsafe or intersect(reachset[i, :], unsafeset)
        if goalreached and not unsafe:
            np.savetxt('systems_with_networks/reachnn_oscillator_benchmark/valid/nn_'+ID+str(it)+'_relu_tanh', control_param+pert)
            # save()
            assert False

        # g1 = W_distance(reachset[-1, :], goalset, printC)
        # s1 = W_distance(reachset[5, :], unsafeset, printC)

        g1 = geometry(reachset[-1, :], goalset, printC)
        s1 = geometry(reachset[5, :], unsafeset, printC)        
        # print(reachset[-1, :])
        # assert False

        np.savetxt('systems_with_networks/reachnn_oscillator_benchmark/nn_test_relu_tanh'+ID, control_param-pert)
        os.system('./benchmark_oscillator_relu_tanh 0.01 20 4 6 0 '+str(NUM))
        reachset = np.load('StepReach0'+ID+'.npy')
        reachset = np.reshape(reachset, (-1, 4))

        goalreached = intersect(reachset[-1, :], goalset)
        unsafe = False
        for i in range(len(reachset)):
            unsafe = unsafe or intersect(reachset[i, :], unsafeset)
        if goalreached and not unsafe:
            np.savetxt('systems_with_networks/reachnn_oscillator_benchmark/valid/nn_'+ID+str(it)+'_relu_tanh', control_param-pert)
            # save()
            assert False

        # g2 = W_distance(reachset[-1, :], goalset, printC)
        # s2 = W_distance(reachset[5, :], unsafeset, printC)

        g2 = geometry(reachset[-1, :], goalset, printC)
        s2 = geometry(reachset[5, :], unsafeset, printC)

        goalgrad[index] = 0.5*(g1-g2)/pert[index]
        safetygrad[index] = 0.5*(s1-s2)/pert[index]

        glist.append(0.5*(g1+g1))
        slist.append(0.5*(s1+s2))

    return goalgrad, safetygrad 

def save():
    np.save('systems_with_networks/reachnn_oscillator_benchmark/geo/data/glist'+ID+'.npy', np.array(glist))
    np.save('systems_with_networks/reachnn_oscillator_benchmark/geo/data/slist'+ID+'.npy', np.array(slist))
    np.save('systems_with_networks/reachnn_oscillator_benchmark/geo/data/time'+ID+'.npy', np.array(timelist))

if __name__ == '__main__':
    goalset = np.array([-0.05, -0.05, 0.05, 0.05])
    unsafeset = np.array([-0.3, 0.2, -0.25, 0.35])
    control_param = get_params()
    # control_param = np.loadtxt('init_control_param_6_tora')
    # assert False

    global glist
    glist = []
    global slist
    slist = []

    timelist = []

    for it in range(60):
        print('------ Here begins ' + str(it) + ' iterations ------')
        start = time.time()
        goalgrad, safetygrad = gradient(it, control_param, goalset, unsafeset)
        print(goalgrad, safetygrad)
        goalgrad = np.clip(goalgrad, -5, 5)
        safetygrad = np.clip(safetygrad, -0.5, 0.5)
        control_param -= 0.5 * goalgrad
        control_param += 0.001 * safetygrad
        timelist.append(time.time()-start)


    
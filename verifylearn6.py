import numpy as np
import os
import wasserstein as was
import torch
import time
import sys

NUM = sys.argv[1]
ID = '_' + NUM

def get_params():
    hidden_size = 3
    all_param = []
    init_param = np.random.rand(6*hidden_size+1)
    init_param = init_param.tolist()

    all_param += init_param
    all_param += [0, 11]
    all_param = [4, 1, 1, hidden_size] + all_param

    all_param[8] = 0
    all_param[13] = 0
    all_param[18] = 0
    all_param[22] = 0

    np.savetxt('systems_with_networks/reachnn_benchmark_6_tora/nn_test_relu_tanh', np.array(all_param))
    return np.array(all_param)


def W_distance(reachset, goalset, printC):

    N = 100
    x0 = np.random.uniform(reachset[0], reachset[4], size=(N,))
    x1 = np.random.uniform(reachset[1], reachset[5], size=(N,))
    # x2 = np.random.uniform(reachset[2], reachset[6], size=(N,))
    # x3 = np.random.uniform(reachset[3], reachset[7], size=(N,))
    # x = np.vstack((x0, x1, x2, x3))
    x = np.vstack((x0, x1))

    y0 = np.random.uniform(goalset[0], goalset[4], size=(N,))
    y1 = np.random.uniform(goalset[1], goalset[5], size=(N,))
    # y2 = np.random.uniform(goalset[2], goalset[6], size=(N,))
    # y3 = np.random.uniform(goalset[3], goalset[7], size=(N,))
    # y = np.vstack((y0, y1, y2, y3))
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
    assert reachset.shape == (8, )
    if reachset[4] <= set2[0] or reachset[0] >= set2[4] or reachset[5] <= set2[1] or reachset[1] >= set2[5]:
        return False    
    else:
        return True

def gradient(it, control_param, goalset, unsafeset):

    goalgrad = np.zeros(len(control_param), )
    safetygrad = np.zeros(len(control_param), )
    delta = np.random.uniform(low=-0.05, high=0.05, size=(control_param.size))
    index_list = [4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21]

    for index in index_list:
        if index <= 5:
            printC = True
        else:
            printC = False 
        pert = np.zeros(len(control_param), )
        pert[index] += delta[index]
        np.savetxt('systems_with_networks/reachnn_benchmark_6_tora/nn_test_relu_tanh', control_param+pert)
        os.system('./benchmark6_relu_tanh 0.01 10 4 6 0 '+str(NUM))
        # assert False
        reachset = np.load('StepReach6'+ID+'.npy')
        reachset = np.reshape(reachset, (-1, 8))

        goalreached = intersect(reachset[-1, :], goalset)
        unsafe = False
        for i in range(len(reachset)):
            unsafe = unsafe or intersect(reachset[i, :], unsafeset)
        if goalreached and not unsafe:
            np.savetxt('systems_with_networks/reachnn_benchmark_6_tora/valid/nn_'+ID+str(it)+'_relu_tanh', control_param+pert)
            assert False

        g1 = W_distance(reachset[-1, :], goalset, printC)
        s1 = W_distance(reachset[4, :], unsafeset, printC)
        # print(reachset[-1, :])
        # assert False

        np.savetxt('systems_with_networks/reachnn_benchmark_6_tora/nn_test_relu_tanh', control_param-pert)
        os.system('./benchmark6_relu_tanh 0.01 10 4 6 0 '+str(NUM))
        reachset = np.load('StepReach6'+ID+'.npy')
        reachset = np.reshape(reachset, (-1, 8))

        goalreached = intersect(reachset[-1, :], goalset)
        unsafe = False
        for i in range(len(reachset)):
            unsafe = unsafe or intersect(reachset[i, :], unsafeset)
        if goalreached and not unsafe:
            np.savetxt('systems_with_networks/reachnn_benchmark_6_tora/valid/nn_'+ID+str(it)+'_relu_tanh', control_param-pert)
            assert False

        g2 = W_distance(reachset[-1, :], goalset, printC)
        s2 = W_distance(reachset[4, :], unsafeset, printC)

        goalgrad[index] = 0.5*(g1-g2)/pert[index]
        safetygrad[index] = 0.5*(s1-s2)/pert[index]

        glist.append(0.5*(g1+g1))
        slist.append(0.5*(s1+s2))

    return goalgrad, safetygrad 


if __name__ == '__main__':
    # goalset = np.array([0., -0.98, -0.6, -1, 0.05, -0.93, -0.4, 1])
    goalset = np.array([0.05, -0.95, -0.6, -1, 0.1, -0.9, -0.4, 1])
    unsafeset = np.array([-0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1])
    # control_param = get_params()
    # np.savetxt('init_control_param_6_tora', control_param)

    control_param = np.loadtxt('init_control_param_6_tora')
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
        control_param -= 0.01 * goalgrad
        # control_param += 0.0005 * safetygrad
        timelist.append(time.time()-start)

    # np.save('systems_with_networks/reachnn_benchmark_6_tora/data/glist'+ID+'.npy', np.array(glist))
    # np.save('systems_with_networks/reachnn_benchmark_6_tora/data/slist'+ID+'.npy', np.array(slist))
    # np.save('systems_with_networks/reachnn_benchmark_6_tora/data/time'+ID+'.npy', np.array(timelist))
    
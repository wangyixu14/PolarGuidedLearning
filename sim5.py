import numpy as np

def relu(x):
    return max(x, 0)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def NN(state):
    assert state.shape == (3, )
    h1 = relu(state.dot(np.array([6.91e-4, 2.22e-1, 4.42e-2])))
    h2 = relu(state.dot(np.array([8.93e-2, 1.56e-1, 9.68e-2])))
    h3 = relu(state.dot(np.array([4.16e-4, -2.74e-2, 3.17e-2])))

    output = tanh(np.array([h1, h2, h3]).dot(np.array([-1.95e-1, -2.66e-1, 1.19e-2])))
    return output

class Benchmark:
    deltaT = 0.05
    u_range = 11
    max_iteration = 48
    def __init__(self, x0=None, x1=None, x2=None):
        if x0 is None or x1 is None:
            x0 = np.random.uniform(low=0.38, high=0.4, size=1)[0]
            x1 = np.random.uniform(low=0.45, high=0.47, size=1)[0]
            x2 = np.random.uniform(low=0.25, high=0.27, size=1)[0]
            self.x0 = x0
            self.x1 = x1
            self.x2 = x2
        else:
            self.x0 = x0
            self.x1 = x1
            self.x2 = x2
        
        self.t = 0
        self.state = np.array([self.x0, self.x1, self.x2])

    def reset(self, x0=None, x1=None):
        if x0 is None or x1 is None:
            x0 = np.random.uniform(low=0.38, high=0.4, size=1)[0]
            x1 = np.random.uniform(low=0.45, high=0.47, size=1)[0]
            x2 = np.random.uniform(low=0.25, high=0.27, size=1)[0]
            self.x0 = x0
            self.x1 = x1
            self.x2 = x2
        else:
            self.x0 = x0
            self.x1 = x1
            self.x2 = x2
        
        self.t = 0
        self.state = np.array([self.x0, self.x1, self.x2])
        return self.state

    def step(self, action):
        u = action * self.u_range
        x0_tmp = self.state[0] + self.deltaT*(self.state[0]**3 - self.state[1])
        x1_tmp = self.state[1] + self.deltaT*(self.state[2])
        x2_tmp = self.state[2] + self.deltaT*u
        
        self.t = self.t + 1
        # reward = self.design_reward()
        self.state = np.array([x0_tmp, x1_tmp, x2_tmp])
        # done = self.if_unsafe() or self.t == self.max_iteration
        return self.state, 0, False

    def design_reward():
        r = 0
        # tarining actor2800 and actor2900
        # actor
        r -= 10 * abs(self.state[0])
        r -= 10 * abs(self.state[1])
        # r -= 0.2 * abs(u)
        # r -= smoothness * abs(u - u_last)
        # if self.if_unsafe():
        #     r -= 50
        # else:
        #     r += 5       
        return r

    def if_unsafe(self):
        if self.state[0] in Interval(-0.3, -0.25) and self.state[1] in Interval(0.2, 0.35):
            return 1
        else:
            return 0

if __name__ == '__main__':
    env = Benchmark()
    state = env.reset()

    for i in range(env.max_iteration):
        control_input = NN(state)
        # print(control_input)
        state, r, _ = env.step(control_input)
        print(state)
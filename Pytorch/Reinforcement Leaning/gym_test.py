import gym
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import LJY_utils
import LJY_visualize_tools


def basic_policy(state):
    #random policy
    angle = state[2]
    return 0 if angle < 0 else 1

class NN_approximator():
    def __init__(self):
        class Policy_network(nn.Module):
            def __init__(self):
                super().__init__()
                self.main = nn.Sequential(
                    nn.Linear(4, 24),
                    nn.ReLU(),

                    nn.Linear(24, 24),
                    nn.ReLU(),

                    nn.Linear(24, 2),
                    nn.Softmax()
                )
                print(self.main)

            def forward(self, input):
                output = self.main(input)
                return output

        self.policy_network = Policy_network()
        self.policy_network.apply(LJY_utils.weights_init)
        self.policy_network.cuda()
        self.loss = nn.MSELoss().cuda()
        self.optimizer = optim.Adam(self.policy_network.parameters(), betas=(0.5, 0.999), lr=3e-4)
        #self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=2e-33)
        #self.optimizer = optim.SGD(self.policy_network.parameters(), lr=2e-3)
        self.epsilon = 1
        self.decay = 0.9999
        self.epsilon_min = 0.01

    def greedy_policy(self, q_value):
        max_idx = np.where(np.asarray(q_value == q_value.max()) == 1)[0].item()
        return max_idx


    def epsilon_greedy_policy(self, q_value):
        if self.epsilon > random.random():
            max_idx = np.where(np.asarray(q_value == q_value.max()) == 1)[0].item()
            return max_idx
        else:
            if 0.5 < random.random():
                return 0
            else:
                return 1
    def epsilon_decay(self):
        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.decay

    def select_action(self,state, off_policy = False):
        with torch.no_grad():
            q_value = self.policy_network(state)
        if off_policy == True:
            action = self.greedy_policy(q_value.detach())
        else :
            action = self.epsilon_greedy_policy(q_value.detach())
        return q_value, action


    def optimize(self, state, action, reward, discount_factor, next_state):

        q_value = self.policy_network(state)
        #print(q_value)
        with torch.no_grad():
            next_q_value, next_action = self.select_action(torch.Tensor(next_state).cuda(),off_policy = True)

        target = q_value.clone().data # keeping same value except to selected action for using MSE/BCE.
        target[action] = reward + discount_factor*next_q_value[next_action]
        #print(reward + discount_factor*next_q_value[next_action])

        #print(q_value)
        #print(target)
        loss = self.loss(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(loss.data)
        return loss.data





env = gym.make("CartPole-v0")
observed_state = env.reset()
print(observed_state)
print(env.action_space)

discount_factor = 0.99

win_dict = LJY_visualize_tools.win_dict()
nn_approximator = NN_approximator()
for epi in range(100000):
    #selected_policy = nn_approximator.policy
    #selected_policy = basic_policy
    epi_reward = 0
    epi_loss = 0
    state = env.reset()
    env.render()
    state = torch.Tensor(state).cuda()
    for iter in range(1000):

        #q_value, action = selected_policy(state)
        q_value, action = nn_approximator.select_action(state.detach())

        next_state, reward, done, info = env.step(action)
        env.render()

        loss = nn_approximator.optimize(state, action, reward, discount_factor, next_state)
        epi_loss += loss
        epi_reward += reward
        if done:
            break
        state = torch.Tensor(next_state).cuda()
    nn_approximator.epsilon_decay()
    epi_reward = np.asarray(epi_reward)
    print('[%d/%d epsilon : %f] epi / iter' % (epi, iter, nn_approximator.epsilon))
    win_dict = LJY_visualize_tools.draw_lines_to_windict(win_dict, [epi_reward, loss/iter*10, 0], ['reward per episode','loss_iter', 'zero'], 0,
                                                         epi, 0)

#rint(np.mean(total_reward), np.std(total_reward), np.min(total_reward), np.max(total_reward))
import gym
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import LJY_utils
import LJY_visualize_tools


def random_policy(state):
    random_action = env.action_space.sample()
    return random_action

def basic_policy(state):
    angle = state[2]
    return 0 if angle < 0 else 1

class NN_approximator():


    def __init__(self):
        class Policy_network(nn.Module):
            def __init__(self):
                super().__init__()
                self.main = nn.Sequential(
                    nn.Linear(4, 4),
                    nn.ReLU(),

                    nn.Linear(4, 4),
                    nn.ReLU(),

                    nn.Linear(4, 2),
                    nn.Sigmoid()
                )
                print(self.main)

            def forward(self, input):
                output = self.main(input)
                return output

        self.BCE_loss = nn.BCELoss().cuda()
        self.policy_network = Policy_network()
        self.policy_network.apply(LJY_utils.weights_init)
        self.policy_network.cuda()




    def deep_SARSA_function(self,state):
        # SARSA
        q_value = self.policy_network(state)
        return q_value

    def epsilon_greedy_policy(self, q_value):
        epslion = 0.01
        if epslion < random.randrange(0,1):
            max_idx = np.where(np.asarray(q_value == q_value.max()) == 1)[0].item()
            return max_idx
        else:
            if 0.5 < random.randrange(0, 1):
                return 0
            else:
                return 1

    def policy(self,state):
        q_value = self.deep_SARSA_function(state)
        action = self.epsilon_greedy_policy(q_value)
        return q_value, action




env = gym.make("CartPole-v0")
observed_state = env.reset()
print(observed_state)
print(env.action_space)



win_dict = LJY_visualize_tools.win_dict()
nn_approximator = NN_approximator()
for epi in range(500):
    selected_policy = nn_approximator.policy
    #selected_policy = basic_policy
    epi_reward = 0
    state = env.reset()
    env.render()

    for iter in range(1000):
        state = torch.Tensor(state).cuda()

        q_value, action = selected_policy(state)

        state, reward, done, info = env.step(action)
        env.render()

        epi_reward += reward
        if done:
            break
    epi_reward = np.asarray(epi_reward)
    print('[%d/%d] epi / iter' % (epi, iter))
    win_dict = LJY_visualize_tools.draw_lines_to_windict(win_dict, [epi_reward, 0], ['reward per episode', 'zero'], 0,
                                                         epi, 0)

print(np.mean(total_reward), np.std(total_reward), np.min(total_reward), np.max(total_reward))
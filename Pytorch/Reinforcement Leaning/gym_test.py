import gym
import numpy as np


def random_action():
    for _ in range(1000):
        obs = env.reset()
        env.render()
        random_action = env.action_space.sample()

        state, reward, done, info = env.step(random_action)
        if done:
            break


def basic_action():
    def basic_policy(state):
        angle = state[2]
        return 0 if angle < 0 else 1

    total_reward = []

    for epi in range(500):
        epi_reward = 0
        state = env.reset()
        env.render()
        for iter in range(1000):
            action = basic_policy(state)
            state, reward, done, info = env.step(action)
            env.render()
            epi_reward += reward
            if done:
                break
        total_reward.append(epi_reward)

    print(np.mean(total_reward), np.std(total_reward), np.min(total_reward), np.max(total_reward))

env = gym.make("CartPole-v0")
observed_state = env.reset()
print(observed_state)
print(env.action_space)

#random_action()
basic_action()


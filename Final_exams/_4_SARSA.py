import numpy as np
import random


def run_action(current_state, current_action):
    # run action with current state and current action
    next_state = current_state
    if current_action == 0:
        next_state = [next_state[0] + 1, next_state[1]]
    elif current_action == 1:
        next_state = [next_state[0] - 1, next_state[1]]
    elif current_action == 2:
        next_state = [next_state[0], next_state[1] + 1]
    elif current_action == 3:
        next_state = [next_state[0], next_state[1] - 1]

    if (next_state[0] >= map_x_min and next_state[0] <= map_x_max and next_state[1] >= map_y_min and next_state[
        1] <= map_y_max):
        reward = map[next_state[0],next_state[1]]

    return next_state, reward


def epsilon_greedy(current_state):
    # epsilon_greedy
    E = Q[current_state[0], current_state[1], 0]
    W = Q[current_state[0], current_state[1], 1]
    N = Q[current_state[0], current_state[1], 2]
    S = Q[current_state[0], current_state[1], 3]
    Q_set = [E, W, N, S]

    No_entry_Q_set = Q_set.copy()


    while True:
        max_value = np.max(No_entry_Q_set)
        candidate_indices = []
        if np.random.binomial(1, epsilon) == 0:
            #0.9
            for (index, value) in enumerate(No_entry_Q_set):
                if value == max_value:
                    candidate_indices.append(index)
        else :
            #0.1
            for (index, value) in enumerate(No_entry_Q_set):
                if -10000 in No_entry_Q_set:
                    if value != max_value:
                        candidate_indices.append(index)
                else:
                    if value != -10000:
                        candidate_indices.append(index)

        selected_action = random.sample(candidate_indices, 1)[0]

        # preventing out-of-range indecies
        next_state = current_state.copy()
        if selected_action == 0:
            next_state = [next_state[0] + 1, next_state[1]]
        elif selected_action == 1:
            next_state = [next_state[0] - 1, next_state[1]]
        elif selected_action == 2:
            next_state = [next_state[0], next_state[1] + 1]
        elif selected_action == 3:
            next_state = [next_state[0], next_state[1] - 1]

        if not(next_state[0] >= map_x_min and next_state[0] <= map_x_max and next_state[1] >= map_y_min
               and next_state[1] <= map_y_max):
            No_entry_Q_set[selected_action] = -10000
            # coloring index caused out of range index
        else:
            break

    return selected_action

epsilon = 0.1
discount_factor = 0.9
learning_rate = 0.01

road = 0
wall = -100
goal = 1

map = np.array([[road,road,goal,wall,road,road,road],
                [road,wall,road,wall,road,wall,wall],
                [road,wall,road,wall,road,road,road],
                [road,wall,road,wall,road,wall,road],
                [road,wall,road,road,road,wall,road],
                [road,wall,road,wall,road,wall,road],
                [road,road,road,wall,road,road,road]])
map_x_max = len(map)-1
map_x_min = 0
map_y_max = len(map)-1
map_y_min = 0

initial_state = [0, 6]
actions = ['E', 'W', 'N', 'S']
Q = np.asarray([[[np.random.normal(0,0.01) for i in range(len(actions))] for j in range(len(map))] for k in range(len(map))])
#Q = np.asarray([[[0 for i in range(len(actions))] for j in range(len(map))] for k in range(len(map))])

total_episode=100000
for episode in range(total_episode):
    Done = False
    current_state = initial_state.copy()
    current_action = epsilon_greedy(current_state)
    # select actions a from s (epsilon greedy)
    while not Done:
        #print(current_state)
        current_Q = Q[current_state[0], current_state[1], current_action].copy()
        action = current_action
        E_Q = Q[current_state[0], current_state[1], 0]
        W_Q = Q[current_state[0], current_state[1], 1]
        N_Q = Q[current_state[0], current_state[1], 2]
        S_Q = Q[current_state[0], current_state[1], 3]
        next_state, reward = run_action(current_state, current_action)
        next_action = epsilon_greedy(next_state)
        next_Q = Q[next_state[0], next_state[1], next_action].copy()
        Q[current_state[0], current_state[1], current_action] = (current_Q + learning_rate*(reward + discount_factor*(next_Q)-current_Q)).copy()
        if reward == 1 or reward == -100:
            Done = True
            if reward == 1:
                message = "Find Goal!"
            elif reward == -100:
                message = "You Died."
        current_state = next_state.copy()
        current_action = next_action
    print("[%d episode] Done! your final position is [%d, %d]. %s"%(episode,next_state[0], next_state[1],message))



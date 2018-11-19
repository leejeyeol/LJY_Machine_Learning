# pseudo code
max_energy = 10000
residual_energy = 10000
class Agent():
    def __init__(self):
    action_probability = policy_network()
    def policy(self, cur_state, residual_ratio):
        action_probability = action_network(cur_state)
        if rand < residual_ratio:
            #exploration
            action = random_policy(action_probability)
        else:
            #exploitation
            action = max_policy(cur_state)
        return action

    def update(self, cur_state, next_state, action, energy):

class Environment():
    def __init__(self):
        self.env_module = "gym_pong"
        self.cur_state = self.get_cur_state()

    def get_cur_state(self):
        cur_state = self.env_module.get_cur_state()
        return cur_state


energy_distributor = Energy_distributor()
agent = Agent()
environment = Environment()

while residual_energy <= 0:
        residual_energy -= 1  # time decay
        residual_ratio = residual_energy/max_energy

        cur_state = environment.get_cur_state()
        action = agent.policy(cur_state, residual_ratio)
        energy = energy_distributor.energy_distribute_by_policy(action, cur_state) # energy = 0 ~ 100

        reward, next_state = environment.step(action)

        energy_distributor.update(cur_state, next_state, action, reward)
        agent.update(cur_state, next_state, action, energy)





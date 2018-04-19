import gym
import gym_lavaland
import numpy as np
import matplotlib.pyplot as plt

# baseline: directly plans with the proxy reward function.
def num_to_coord(num, WORLD_WIDTH):
    r, c = num//WORLD_WIDTH, num %WORLD_WIDTH
    return (r, c)

def coord_To_Num(coord, WORLD_WIDTH):
    r, c = coord[0], coord[1]
    return r*WORLD_WIDTH + c # num from 0 to 99

def eplisonGreedy(s, Q, eplison, WORLD_WIDTH=10):
    if np.random.rand() < eplison:
        # action_n = 4
        action = np.random.randint(4)
    else:
        rewardSpace = Q[s, :]
        maxReward = np.max(rewardSpace)
        candidateActions = []
        for a, r in enumerate(rewardSpace):
            print("state:", s, " maxReward:", maxReward, " r:", r, " rewardSpace:", rewardSpace)
            if r == maxReward:
                candidateActions.append(a)

        # candidateActions = [a for a, r in enumerate(rewardSpace) if r == maxReward]
        # r, c = num_to_coord(s, WORLD_WIDTH)
    #     if r == 0:
    #         candidateActions.remove(0)
    #     if r == 9:
    #         candidateActions.remove(1)
    #     if c == 0:
    #         candidateActions.remove(2)
    #     if c == 9:
    #         candidateActions.remove(3)
    # #     print(candidateActions)
        action = np.random.choice(candidateActions)
    return action

'''
DYNA-Q
integrates both direct RL and model learning. Planning is one-step tabular q-planning
learning is one-step tabular q-learning. Model improvement from real experience and 
value function/policy improve from direct RL.

model: the model storing for each s,a pair the reward and next state
it is |S| * |A|, model[s_idx][a_idx] = [r, s'_idx]

Q: action-value function |S| * |A|. stores the q value for each possible s,a pair. Each state is its index location
in the flattened maze, thus to get the q value for s, a = qtable[s, a]

S: each player's possible state in our environment (i.e.maze) is the player's location(index)
   in the fattened maze
A: # left, up, right, down = ['L', 'U', 'R', 'D']
'''

# note here states = location in the gridworld
def dyna_q(env, state_n, action_n, init_player_state, WORLD_WIDTH, plan_count=5, n=3000, eplison=0.1, alpha=0.1, discount=0.95):
    # parames
    steps = 0
    observedStates = []
    observedActions = []
    totalReward = np.zeros(n + 1)

    # init Q(s,a) model(s,a) for all s and a
    # by setting random init values to Q tends to give better reproduced result
    Q = np.random.rand(state_n, action_n)  # np.zeros((state_n,action_n))
    reward_model = np.zeros((state_n, action_n))
    nextS_model = np.zeros((state_n, action_n))

    # 1. learning step: from real experience to build imaginary model
    cur_state = init_player_state
    cur_state = coord_To_Num(cur_state, WORLD_WIDTH)
    observedStates.append(cur_state)

    # cannot loop forever, use a max iteration n to guard loop
    time_step = 0
    while time_step < n:
        chosen_act = eplisonGreedy(cur_state, Q, eplison)
        observedActions.append(chosen_act)
        #     print(chosen_act)
        observedActions = list(set(observedActions))
        #     print("actions", observedActions)
        #     observedActions = observedActions.unique()
        done, phi_epsilon, pos, reward = env.step(chosen_act)
        #     print(chosen_act, reward)
        # change dtype to int; otherwise somehow program runs with index as float type error
        next_state = pos
        next_state = coord_To_Num(next_state, WORLD_WIDTH)
        observedStates.append(next_state)
        #     observedStates = observedStates.unique()
        observedStates = list(set(observedStates))
        #     print("states", observedStates)

        # Q value update
        #     print(reward, gamma)

        derta = alpha * (reward + discount * np.max(Q[next_state, :]) - Q[cur_state, chosen_act])
        print("\n", cur_state, next_state, Q[cur_state, chosen_act], derta)
        Q[cur_state, chosen_act] += derta
        print("\n", Q[cur_state, chosen_act])
        # model update
        reward_model[cur_state, chosen_act] = reward
        nextS_model[cur_state, chosen_act] = next_state

        cur_state = next_state

        # 2. planning step: make use of the imagining model in parallal
        for i in range(plan_count):
            sample_state = np.random.choice(observedStates)
            sample_action = np.random.choice(observedActions)
            sample_r = reward_model[sample_state, sample_action]
            sample_next_state = int(nextS_model[sample_state, sample_action])
            Q[sample_state, sample_action] += alpha * (
                        sample_r + discount * np.max(Q[sample_next_state, :]) - Q[sample_state, sample_action])

        time_step = time_step + 1
        totalReward[time_step] = totalReward[time_step - 1] + reward
        #     print(time_step, totalReward)

        # if the game has just terminated before the player got a chance to do anything!,
        # `discount` will be 0.0
        if done:
            # cur_state = init_player_state
            cur_state = coord_To_Num(init_player_state, WORLD_WIDTH)

    return totalReward

if __name__ == "__main__":
    # plan_counts = [5, 50, 200]
    plan_counts = [50]
    training = True
    # index 0 = dirt  index 1 = grass index 2 = terminal
    # training stage: missing index 3 = lava?
    # testing stage: index 3 = lava
    if training:
        proxy_reward = np.array([-1, -5, 10, 0])
    # else:
    #     proxy_reward = np.array([-1, -5, 10])

    rewards_list = []

    for pc in plan_counts:
      rewards = []
      for i in range(20): #average on 20 experiments
        # game = make_game()
        # obs, reward, gamma = game.its_showtime()
        env = gym.make('Simple_training_lavaland-v0')
        pos = env.reset(proxy_reward)
        WORLD_HEIGHT, WORLD_WIDTH = env.getDimension()
        state_n = WORLD_HEIGHT * WORLD_WIDTH
        action_n = env.getActionCount()
        init_player_state = pos
        reward = dyna_q(env, state_n, action_n, init_player_state, WORLD_WIDTH, plan_count=pc, n=3000, eplison=0.1, alpha=0.1, discount=0.95)
        rewards.append(reward)
      rewards = np.array(rewards)
      rewards_list.append(rewards)

    rewards_list = np.array(rewards_list)
    for idx, rewards in enumerate(rewards_list):
      plt.figure()
      plt.plot(rewards.mean(axis=0), label='Dyna-Q, planning-loop-count-{}'.format(plan_counts[idx]))
      plt.xlabel('Time Steps')
      plt.ylabel('Cumulative reward')
      plt.legend()
    plt.show()

import gym
import gym_lavaland
import numpy as np
import time
import matplotlib.pyplot as plt

TMAX = 2000
EPISILON=0.2
LR=0.01
DISCOUNT=0.95
PLANCOUNT=50
ITERMAX = 1000
PLOT_FIGURE = True
PRINTWORLD = False

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
            # print("state:", s, " maxReward:", maxReward, " r:", r, " rewardSpace:", rewardSpace)
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

def run_test_exp(env, proxy_reward, policy=None, discount=DISCOUNT, render=False):
    pos = env.reset(proxy_reward)
    WORLD_HEIGHT, WORLD_WIDTH = env.getDimension()
    total_reward = 0
    optimal_traj = []
    for step_idx in range(TMAX):
        if render:
            env.render()
        if policy is None:
            action = np.random.randint(4)
        else:
            # print("step [{}] at pos [{}, {}]".format(step_idx, pos[0], pos[1]))
            optimal_traj.append(pos)
            cur_state = coord_To_Num(pos, WORLD_WIDTH)
            action = policy[cur_state]
        done, phi_epsilon, pos, reward = env.step(action)
        total_reward += reward
        # total_reward += discount ** step_idx * reward #bellman
        if done:
            optimal_traj.append(pos)
            # print("step [{}] at pos [{}, {}] hit the GOLD with reward [{}]!".format(step_idx, pos[0], pos[1], total_reward))
            break
    return total_reward, phi_epsilon, optimal_traj

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
def dyna_q(iter, Q, reward_model, nextS_model, env, proxy_reward, WORLD_WIDTH, scr=None, plan_count=5, n=3000, eplison=0.1, alpha=0.01, discount=0.95):

    observedStates = []
    observedActions = []
    totalReward = np.zeros(n + 1)

    # init Q(s,a) model(s,a) for all s and a
    # by setting random init values to Q tends to give better reproduced result
    # Q = np.zeros((state_n,action_n)) #np.random.rand(state_n, action_n)
    # reward_model = np.zeros((state_n, action_n))
    # nextS_model = np.zeros((state_n, action_n))

    # 1. learning step: from real experience to build imaginary model
    pos = env.reset(proxy_reward)
    cur_state = pos
    cur_state = coord_To_Num(cur_state, WORLD_WIDTH)
    observedStates.append(cur_state)

    # cannot loop forever, use a max iteration n to guard loop
    for time_step in range(n):
        chosen_act = eplisonGreedy(cur_state, Q, eplison)
        observedActions.append(chosen_act)
        observedActions = list(set(observedActions))
        done, phi_epsilon, pos, reward = env.step(chosen_act)
        # change dtype to int; otherwise somehow program runs with index as float type error
        next_state = pos
        next_state = coord_To_Num(next_state, WORLD_WIDTH)
        observedStates.append(next_state)
        observedStates = list(set(observedStates))
        Q[cur_state, chosen_act] += alpha * (reward + discount * np.max(Q[next_state, :]) - Q[cur_state, chosen_act])
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

        totalReward[time_step] = totalReward[time_step - 1] + reward
        # if the game has just terminated before the player got a chance to do anything!,
        # `discount` will be 0.0

        if scr:
            scr, last_row = env.printWorld(scr)
            if done:
                status_update = "exp FINISH at time_step: {} with totalReward: {}".format(time_step, totalReward[time_step])
                scr.addstr(last_row + 1, 0, status_update)
                scr.refresh()
                time.sleep(1)
                break
            else:
                status_update = "exp continues at time_step: {} with totalReward: {}".format(time_step, totalReward[time_step])
                scr.addstr(last_row + 1, 0, status_update)
                scr.refresh()
                time.sleep(0.1)
        else:
            if done:
                # print("iter [{}] step [{}] hit the GOLD with reward [{}]!".format(iter, time_step, totalReward[time_step]))
                break
    return totalReward, Q, phi_epsilon, time_step

def printTraj(traj):
    if traj:
        totalStep = len(traj)
        traj_str = ""
        for step, coor in enumerate(traj):
            if step + 1 == totalStep:
                traj_str += "({}, {})".format(coor[0], coor[1])
            else:
                traj_str += "({}, {}) -> ".format(coor[0], coor[1])
        print(traj_str)

def agent_learn(proxy_reward, env, training_env):
    WORLD_HEIGHT, WORLD_WIDTH = env.getDimension()
    state_n = WORLD_HEIGHT * WORLD_WIDTH
    action_n = env.getActionCount()

    Q = np.zeros((state_n, action_n))
    reward_model = np.zeros((state_n, action_n))
    nextS_model = np.zeros((state_n, action_n))

    rewards_iter = []
    for i in range(ITERMAX):  # average on ITERMAX experiments (iteration)
        if PRINTWORLD:
            import curses, time

            scr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            statusString = "exp: {} START, reward: {}".format(i, '0')
            scr.addstr(0, 0, statusString)

        if PRINTWORLD:
            reward_iter_i, Q, phi_traj, ts = dyna_q(i, Q, reward_model, nextS_model, env, proxy_reward, WORLD_WIDTH,
                                                    scr, plan_count=PLANCOUNT, n=TMAX, eplison=EPISILON, alpha=LR,
                                                    discount=DISCOUNT)
            curses.nocbreak()
            scr.keypad(False)
            curses.echo()
        else:
            reward_iter_i, Q, phi_traj, ts = dyna_q(i, Q, reward_model, nextS_model, env, proxy_reward, WORLD_WIDTH,
                                                    plan_count=PLANCOUNT, n=TMAX, eplison=EPISILON, alpha=LR,
                                                    discount=DISCOUNT)
        rewards_iter.append(reward_iter_i)

    rewards_iter = np.array(rewards_iter)
    # print("frac_traj_hit_lava baseline = ", traj_hit_lava_count/ITERMAX)

    if PLOT_FIGURE:
        plt.figure()
        jump = ITERMAX // 5
        for idx, iter in enumerate(rewards_iter[0::jump]):
            plt.plot(iter, label='iteration {}'.format(str((idx + 1) * jump)))
        plt.xlabel('timestemp')
        plt.ylabel('Total reward per iteration')
        plt.legend()
        plt.show()

    # after all experiments (iterations)
    # get optimal policy
    solution_policy = np.argmax(Q, axis=1)
    solution_policy_scores, solution_policy_traj_phi, solution_traj = run_test_exp(env, proxy_reward, solution_policy,
                                                                                   DISCOUNT,
                                                                                   False)
    if training_env:
        print("Training env optimal trajectory: ")
    else:
        print("Testing env optimal trajectory: ")
    printTraj(solution_traj)


if __name__ == "__main__":
    # part 1: dyna-q (baseline on training env -- figure 2 left)
    proxy_reward = np.array([-0.1, -0.5, 10, 0])
    # make env
    env = gym.make('Simple_training_lavaland-v0')
    agent_learn(proxy_reward,env,True)

    # part 2: dyna-1 (baseline on testing env -- figure 2 middle)
    # make env
    env_test = gym.make('Simple_testing_lavaland-v0')
    agent_learn(proxy_reward,env_test,False)

#
#
#
#
# if __name__ == "__main__":
#
#     printWorld = False
#     traj_hit_lava_count = 0
#     conv_reward = None
#     conv_test_reward = None
#     conv_train_iter = 0
#     conv_train_iter_found = False
#     conv_test_iter = 0
#     conv_test_iter_found = False
#     solution_policy = None
#     # plan_counts = [5, 50, 200]
#     # plan_counts = 50
#     # training = True
#     # index 0 = dirt  index 1 = grass index 2 = terminal
#     # training stage: missing index 3 = lava?
#     # testing stage: index 3 = lava
#
#     proxy_reward = np.array([-0.1, -0.5, 10, 0])
#     rewards_plan_count = []
#     # make env
#     env = gym.make('Simple_training_lavaland-v0')
#     WORLD_HEIGHT, WORLD_WIDTH = env.getDimension()
#     state_n = WORLD_HEIGHT * WORLD_WIDTH
#     action_n = env.getActionCount()
#
#     Q = np.zeros((state_n, action_n))
#     reward_model = np.zeros((state_n, action_n))
#     nextS_model = np.zeros((state_n, action_n))
#
#     rewards_iter = []
#     for i in range(ITERMAX): #average on 20 experiments (iteration)
#         if printWorld:
#             import curses, time
#             scr = curses.initscr()
#             curses.noecho()
#             curses.cbreak()
#             statusString = "exp: {} START, reward: {}".format(i, '0')
#             scr.addstr(0, 0, statusString)
#
#         # env = gym.make('Simple_training_lavaland-v0')
#         # pos = env.reset(proxy_reward)
#         # WORLD_HEIGHT, WORLD_WIDTH = env.getDimension()
#         # state_n = WORLD_HEIGHT * WORLD_WIDTH
#         # action_n = env.getActionCount()
#         # init_player_state = pos
#
#         if printWorld:
#             reward_iter_i, Q, phi_traj, ts = dyna_q(i, Q, reward_model, nextS_model, env, proxy_reward, WORLD_WIDTH, scr, plan_count=PLANCOUNT, n=TMAX, eplison=EPISILON, alpha=LR, discount=DISCOUNT)
#             curses.nocbreak()
#             scr.keypad(False)
#             curses.echo()
#         else:
#             reward_iter_i, Q, phi_traj, ts = dyna_q(i, Q, reward_model, nextS_model, env, proxy_reward, WORLD_WIDTH, plan_count=PLANCOUNT, n=TMAX, eplison=EPISILON, alpha=LR, discount=DISCOUNT)
#         rewards_iter.append(reward_iter_i)
#
#         # if solution_policy:
#         #     new_sp = np.argmax(Q, axis=1)
#         #     if new_sp == solution_policy:
#         #         # converged - optimal policy found, change environment
#         #         print("ENV CHANGED AT ITERATION ", i)
#         #         conv_train_iter = i
#         #         env.changeEnv()
#         # else:
#         #     solution_policy = np.argmax(Q, axis=1)
#
#         # print("phi: ", phi_traj[2], phi_traj[3])
#
#         # if phi_traj[2] != 0 and phi_traj[3] == 0.0: # training, hit gold
#         #     if conv_reward and not conv_train_iter_found:
#         #         # print(conv_reward, reward_iter_i[ts])
#         #         if conv_reward == reward_iter_i[ts]:
#         #             # converged - optimal policy found, change environment
#         #             print("ENV CHANGED AT ITERATION ", i)
#         #             conv_train_iter = i
#         #             conv_train_iter_found = True
#         #             env.changeEnv()
#         #         elif conv_reward < reward_iter_i[ts]:
#         #             conv_reward = reward_iter_i[ts]
#         #     elif not conv_train_iter_found:
#         #         conv_reward = reward_iter_i[ts]
#         # elif phi_traj[3] != 0 and phi_traj[2] != 0:
#         #     if conv_test_reward and not conv_test_iter_found:
#         #         if conv_test_reward == reward_iter_i[ts]:
#         #             # test converged - optimal policy found
#         #             conv_test_iter_found = True
#         #             print("TEST ENV ADAPTED AT ITERATION [", i, "], used [", i - conv_train_iter,"] iterations")
#         #         elif conv_test_reward < reward_iter_i[ts]:
#         #             conv_test_reward = reward_iter_i[ts]
#         #     elif not conv_test_iter_found:
#         #         conv_test_reward = reward_iter_i[ts]
#
#     rewards_iter = np.array(rewards_iter)
#     # print("frac_traj_hit_lava baseline = ", traj_hit_lava_count/ITERMAX)
#
#     if PLOT_FIGURE:
#         plt.figure()
#         jump = ITERMAX//5
#         for idx, iter in enumerate(rewards_iter[0::jump]):
#             plt.plot(iter, label='iteration {}'.format(str((idx+1)*jump)))
#         plt.xlabel('timestemp')
#         plt.ylabel('Total reward per iteration')
#         plt.legend()
#         plt.show()
#
#     # # plt.plot(rewards_iter[:, -1], label='Dyna-Q, planning-loop-count-{}'.format(plan_counts))
#     # plt.xlabel('iteration')
#     # plt.ylabel('Total reward per iteration')
#     # plt.legend()
#     # plt.show()
#     # rewards_plan_count.append(rewards_iter)
#     # rewards_plan_count = np.array(rewards_plan_count)
#
#     # after all experiments (iterations)
#     # 1. get optimal policy
#     solution_policy = np.argmax(Q, axis=1)
#     solution_policy_scores_list = []
#     solution_policy_traj_phi_list = []
#     for _ in range(100):
#         solution_policy_scores, solution_policy_traj_phi = run_test_exp(env, proxy_reward, solution_policy, DISCOUNT, False)
#         solution_policy_scores_list.append(solution_policy_scores)
#         solution_policy_traj_phi_list.append(solution_policy_traj_phi)
#     # [solution_policy_scores, solution_policy_traj_phi] = [run_test_exp(env, proxy_reward, solution_policy, DISCOUNT, False) for _ in range(100)]
#
#     print("Average score of solution = ", np.mean(solution_policy_scores_list))
#     # dirt, grass, terminal, lava
#     # count_traj_lava = 0
#     # total_traj = 100
#     # for traj in solution_policy_traj_phi_list:
#     #     if traj[3] != 0:
#     #         count_traj_lava += 1
#     # frac_traj_hit_lava = count_traj_lava / total_traj
#     # print("frac_traj_hit_lava w\o LAVA = ", frac_traj_hit_lava)
#
#     # 2. experiment on lava env
#     # The proxy does not penalize lava, so optimizing it makes the agent go straight through (gray).
#
#     # env = gym.make('Simple_testing_lavaland-v0')
#     # # [solution_policy_scores, solution_policy_traj_phi] = [run_test_exp(env, proxy_reward, solution_policy, DISCOUNT, False) for _ in range(100)]
#     #
#     # solution_policy_scores_list = []
#     # solution_policy_traj_phi_list = []
#     # for _ in range(100):
#     #     solution_policy_scores, solution_policy_traj_phi = run_test_exp(env, proxy_reward, solution_policy, DISCOUNT, False)
#     #     solution_policy_scores_list.append(solution_policy_scores)
#     #     solution_policy_traj_phi_list.append(solution_policy_traj_phi)
#     #
#     # print("Average score of solution with LAVA = ", np.mean(solution_policy_scores_list))
#     # count_traj_lava = 0
#     # total_traj = 100
#     # for traj in solution_policy_traj_phi_list:
#     #     if traj[3] != 0:
#     #         count_traj_lava += 1
#     # frac_traj_hit_lava = count_traj_lava / total_traj
#     # print("frac_traj_hit_lava with LAVA = ", frac_traj_hit_lava)
#
#     # MEANING-LESS
#     # # 2. plot Cumulative reward
#     # # for idx, rewards in enumerate(rewards_plan_count):
#     # plt.figure()
#     # plt.plot(rewards.mean(axis=0), label='Dyna-Q, planning-loop-count-{}'.format(plan_counts))
#     # plt.xlabel('Time Steps')
#     # plt.ylabel('Cumulative reward')
#     # plt.legend()
#     # plt.show()

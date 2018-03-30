import gym
from gym.envs.registration import register
import numpy as np

register(
    id='BoyanChainEnv-v0',
    entry_point='gym.envs.algorithmic:BoyanChainEnv',
    max_episode_steps=200,
)
env = gym.make('BoyanChainEnv-v0')

Phi = [np.array([0,0,0,0]), np.array([0,0,1/4,3/4]), np.array([0,0,1/2,1/2]), np.array([0,0,3/4,1/4]),
       np.array([0,0,1,0]), np.array([0,1/4,3/4,0]), np.array([0,1/2,1/2,0]), np.array([0,3/4,1/4,0]),
       np.array([0,1,0,0]), np.array([1/4,3/4,0,0]), np.array([1/2,1/2,0,0]), np.array([3/4,1/4,0,0]),
       np.array([1,0,0,0])]
num_states = len(Phi)
num_features = len(Phi[0])
num_eps = 50
lambda_lstd = 0.2
A = np.zeros((num_features, num_features))
b = np.zeros((num_features, 1))
beta_opt =np.array([-24,-16,-8,0]).reshape(num_features,1)
result = np.zeros((num_eps))

for i_episode in range(num_eps):
    env.reset()
    cur_state = num_states - 1
    z = Phi[cur_state].reshape((num_features, 1))
    for t in range(100):
        if cur_state == 1:
            action = 0
        else:
            action = env.action_space.sample() #we can take a sample here because an action is chosen with prob 50%
        observation, reward, done, info = env.step(action)
        cur_state = observation['cur_state']
        prev_state = observation['prev_state']
        A = A + z@(Phi[prev_state]-Phi[cur_state]).reshape((1,num_features))
        b = b + reward*z
        z = lambda_lstd*z + Phi[cur_state].reshape((num_features, 1))
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    beta = np.linalg.solve(A,b)
    result[i_episode] = np.linalg.norm(beta-beta_opt,ord=np.inf)

print('hi')
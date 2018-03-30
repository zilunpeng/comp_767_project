import gym
from gym.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt

register(
    id='FiveStateEnv-v0',
    entry_point='gym.envs.algorithmic:FiveStateEnv',
    max_episode_steps=500,
)
env = gym.make('FiveStateEnv-v0')

Phi = [np.array([74.29, 34.61, 73.48, 53.29, 7.79]),
       np.array([61.6, 48.07, 34.68, 36.19, 82.02]),
       np.array([97, 4.88, 8.51, 87.89, 5.17]),
       np.array([41.1, 40.13, 64.63, 92.67, 31.09]),
       np.array([7.76, 79.82, 43.78, 8.56, 61.11])]
num_states = len(Phi)
num_features = len(Phi[0])
num_steps = 100
lambda_lstd = 0.8
gamma = 0.9
A = np.zeros((num_features, num_features))
b = np.zeros((num_features, 1))
theta_opt =np.array([3.916,9.374,-0.933,3.203,-1.093]).reshape(num_features,1)
result = np.zeros((num_steps))

cur_state = env.reset()
z = Phi[cur_state].reshape((num_features, 1))
for t in range(num_steps):
    observation, reward, done, info = env.step(0)
    cur_state = observation['cur_state']
    prev_state = observation['prev_state']
    A = A + z@(Phi[prev_state]-gamma*Phi[cur_state]).reshape((1,num_features))
    b = b + reward*z
    z = lambda_lstd*z + Phi[cur_state].reshape((num_features, 1))
    #env.render()
    theta = np.linalg.solve(A, b)
    result[t] = np.linalg.norm(theta - theta_opt, ord=np.inf)
plt.plot(result)
plt.show()
import gym
from gym.envs.registration import register

register(
    id='BoyanChainEnv-v0',
    entry_point='gym.envs.algorithmic:BoyanChainEnv',
    max_episode_steps=200,
)

env = gym.make('BoyanChainEnv-v0')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render()
from gym.envs.registration import register

register(
    id='Simple_training_lavaland-v0',
    entry_point='gym_lavaland.envs:Simple_training_lavaland',
)

register(
    id='Simple_testing_lavaland-v0',
    entry_point='gym_lavaland.envs:Simple_testing_lavaland',
)

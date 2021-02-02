import gym

gym.envs.register(
     id='random-v0',
     entry_point='agent.enviornment.random:RandomEnv',
     max_episode_steps=20,
)
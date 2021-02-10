import gym

gym.envs.register(
     id='random-v0',
     entry_point='agent.environment.random:RandomEnv',
     max_episode_steps=20,
)
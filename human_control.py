import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper

if __name__ == '__main__':
    env_name = 'FrankaKitchen-v1'
    max_episode_steps = 500

    task = 'microwave'
    task_no_spaces = task.replace(" ", "_")  # 그냥 task 이름들에 space 없애는 것. 별 뜻 없음

    env = gym.make(env_name, max_episode_steps=max_episode_steps,
                   tasks_to_complete=[task], render_mode='human', autoreset=False)

    env = RoboGymObservationWrapper(env, goal=task)

    print(env.env.env.env.env.model.opt.gravity)

    state, _ = env.reset()
    print(state)

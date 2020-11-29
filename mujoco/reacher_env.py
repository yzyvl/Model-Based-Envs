import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dir_path, 'assets/reacher.xml'), 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        # print(vec)
        # exit()
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def mb_step(self, obs, a, next_obs):
        reward_dist = -np.linalg.norm(obs[-3:])
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        return reward, False

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()


if __name__ == '__main__':
    from gym import register
    import gym

    register(id='reacher-v2', entry_point=ReacherEnv, max_episode_steps=50)
    env = gym.make('reacher-v2')

    state = env.reset()
    total = 0
    v_t = 0

    for i in range(50):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        r, done_ = env.mb_step(state, action, next_state)
        v_t += r

        state = next_state

        total += reward
        if done:
            print(i)
            exit()

        print(total, v_t)

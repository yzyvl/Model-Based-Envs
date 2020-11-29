import os

import numpy as np

from gym import utils
from gym import register
from gym.envs.mujoco import mujoco_env


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/walker2d.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((xposafter - xposbefore) / self.dt)
        reward += alive_bonus
        reward -= 0.005 * np.square(a).sum()
        done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def mb_step(self, obs, a, next_obs):
        reward = (next_obs[0] - obs[0]) / self.dt + 1.
        reward -= 0.005 * np.square(a).sum()
        done = not (0.8 < next_obs[1] < 2.0 and -1.0 < next_obs[2] < 1.0)

        return reward, done

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos.flat,
                               self.sim.data.qvel.flat,
                               self.get_body_com("torso").flat])

        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        # return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


if __name__ == '__main__':
    from gym import register
    import gym

    register(id='walker2d-v2', entry_point=Walker2dEnv, max_episode_steps=1000)
    env = gym.make('walker2d-v2')

    state = env.reset()
    total = 0
    v_t = 0

    for i in range(1000):
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
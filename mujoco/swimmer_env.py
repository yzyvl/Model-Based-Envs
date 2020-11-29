import numpy as np
import os

from gym import utils
from gym import register
from gym.envs.mujoco import mujoco_env


# class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#         mujoco_env.MujocoEnv.__init__(self, '%s/assets/swimmer.xml' % dir_path, 5)
#         utils.EzPickle.__init__(self)
#
#     def step(self, a):
#         self.do_simulation(a, self.frame_skip)
#         bounds = self.model.actuator_ctrlrange
#         lb, ub = bounds[:, 0], bounds[:, 1]
#         scaling = (ub - lb) * 0.5
#         ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(a / scaling))
#         reward_fwd = self.data.get_body_xvelr('torso')[0]
#         reward = reward_fwd - ctrl_cost
#         done = False
#         ob = self._get_obs()
#
#         return ob, reward, done, dict(reward_fwd=reward_fwd, reward_ctrl=ctrl_cost)
#
#     def mb_step(self, obs, a, next_obs):
#         bounds = self.model.actuator_ctrlrange
#         lb, ub = bounds[:, 0], bounds[:, 1]
#         scaling = (ub - lb) * 0.5
#         ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(a / scaling))
#         reward_fwd = next_obs[-1]
#         reward = reward_fwd - ctrl_cost
#         done = False
#         return reward, done
#
#     def _get_obs(self):
#         qpos = np.squeeze(self.sim.data.qpos)
#         qvel = np.squeeze(self.sim.data.qvel)
#         # print(qpos[2:5], qvel[2:5], self.get_body_com("torso").flat[:2], self.data.get_body_xvelr('torso').flat[:1])
#         return np.concatenate([self.get_body_com("torso").flat[:2],
#                                qpos[2:5], qvel[2:5],
#                                self.data.get_body_xvelr('torso').flat[:1]])
#
#     def reset_model(self):
#         self.set_state(
#             self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
#             self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
#         )
#         return self._get_obs()


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/swimmer.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def mb_step(self, obs, a, next_obs):
        reward_fwd = (next_obs[0] - obs[0]) / self.dt
        reward_ctrl = - 0.0001 * np.square(a).sum()
        reward = reward_fwd + reward_ctrl

        return reward, False

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()


if __name__ == '__main__':
    from gym import register
    import gym

    register(id='swimmer-v2', entry_point=SwimmerEnv, max_episode_steps=1000)
    env = gym.make('swimmer-v2')

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
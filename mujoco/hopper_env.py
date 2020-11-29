import numpy as np
import os

from gym import utils
from gym import register
from gym.envs.mujoco import mujoco_env


# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/hopper.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}
        #     def state_vector(self):
        #         return np.concatenate([
        #             self.sim.data.qpos.flat,
        #             self.sim.data.qvel.flat
        #         ])
        # self.do_simulation(a, self.frame_skip)
        # ob = self._get_obs()
        # bounds = self.model.actuator_ctrlrange
        # lb, ub = bounds[:, 0], bounds[:, 1]
        # scaling = (ub - lb) * 0.5
        #
        # ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(a / scaling))
        #
        # vel = ob[5]
        # # height_cost = 10*np.maximum(0.45 - ob[0], 0)
        # # ang_cost = 10*np.maximum(abs(ob[1]) - .2, 0)
        # reward = vel - ctrl_cost
        # done = False
        #
        # return ob, reward, done, {}

    # Consist of 11 dimensions
    # 0: z-com
    # 1: forward pitch along y-axis
    # 5: x-comvel
    # 6: z-comvel
    def mb_step(self, obs, a, next_obs):
        reward = (next_obs[0] - obs[0]) / self.dt + 1.
        reward -= 1e-3 * np.square(a).sum()
        done = not (np.isfinite(next_obs).all() and (np.abs(next_obs[2:]) < 100).all() and (next_obs[1] > .7) and (abs(next_obs[2]) < .2))

        return reward, done

    def _get_obs(self):

        return np.concatenate([self.sim.data.qpos.flat,
                               np.clip(self.sim.data.qvel.flat, -10, 10),
                               self.data.get_body_xvelr("torso")[[0, 2]].flat]
                              )

        # return np.concatenate([self.get_body_com("torso")[2].flat,
        #                        self.sim.data.qpos[2:].flat,
        #                        self.data.get_body_xvelr("torso")[[0, 2]].flat,
        #                        self.sim.data.qvel[2:].flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


if __name__ == '__main__':
    from gym import register
    import gym

    register(id='hopper-v2', entry_point=HopperEnv, max_episode_steps=1000)
    env = gym.make('hopper-v2')

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

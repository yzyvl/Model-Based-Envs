import numpy as np
import os
from gym import utils
from gym import register
from gym.envs.mujoco import mujoco_env


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return np.sum(mass * xpos, 0) / np.sum(mass)


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/humanoid.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        pos_before = np.copy(mass_center(self.model, self.sim))[0]
        self.do_simulation(a, self.frame_skip)
        pos_after = np.copy(mass_center(self.model, self.sim))[0]
        alive_bonus = 0.2
        # data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.5 * 1e-3 * np.square(a).sum()
        # quad_impact_cost = .005 * np.square(data.cfrc_ext).sum()
        # quad_impact_cost = min(quad_impact_cost, 10)
        quad_impact_cost = 0
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def mb_step(self, obs, a, next_obs):
        lin_vel_cost = 1.25 * (next_obs[-3] - obs[-3]) / self.dt
        alive_bonus = 0.2
        ctrl_cost = 0.5 * 1e-3 * np.square(a).sum()
        reward = lin_vel_cost - ctrl_cost + alive_bonus
        done = bool((next_obs[2] < 1.0) or (next_obs[2] > 2.0))

        return reward, done

    def _get_obs(self):
        # print(mass_center(self.model, self.sim))
        # exit()
        data = self.sim.data
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat,
                               # data.qfrc_actuator.flat,
                               # np.clip(data.cfrc_ext, -1, 1).flat,
                               self.get_body_com("torso").flat,
                               np.copy(mass_center(self.model, self.sim))  # 3
                               ])

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv, )
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


if __name__ == '__main__':
    from gym import register
    import gym

    register(id='humanoid-v2', entry_point=HumanoidEnv, max_episode_steps=1000)
    env = gym.make('humanoid-v2')

    state = env.reset()

    total = 0
    v_t = 0

    for i in range(1000):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        r, done_ = env.mb_step(state, action, next_state)
        v_t += r

        # print(state)
        # print(next_state)
        # exit()

        state = next_state

        total += reward
        if done:
            print(i)
            exit()

        print(total, v_t)

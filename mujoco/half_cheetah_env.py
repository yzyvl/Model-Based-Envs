import numpy as np
import os

from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_x_torso, self.x_torso = None, None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.prev_x_torso = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        self.x_torso = self.get_body_com("torso")[0]
        reward_ctrl = -0.1 * np.square(a).sum()
        reward_run = (self.x_torso - self.prev_x_torso) / self.dt
        reward = reward_run + reward_ctrl

        ob = self._get_obs()
        done = False

        return ob, reward, done, {}

    def mb_step(self, obs, a, next_obs):
        forward_reward = (next_obs[0] - obs[0]) / self.dt
        ctrl_reward = 0.1 * np.square(a).sum()
        rewards = forward_reward - ctrl_reward

        return rewards, False

    def _get_obs(self):
        z_position = self.sim.data.qpos.flat[1:2]
        y_rotation = self.sim.data.qpos.flat[2:3]
        other_positions = self.sim.data.qpos.flat[3:]
        velocities = self.sim.data.qvel.flat

        # x_torso = np.copy(self.get_body_com("torso")[0:1])
        average_velocity = self.get_body_com("torso").flat[:1]
        y_rotation_sin, y_rotation_cos = np.sin(y_rotation), np.cos(y_rotation)

        obs = np.concatenate([average_velocity, z_position, y_rotation_sin,
                              y_rotation_cos, other_positions, velocities])

        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55


if __name__ == '__main__':
    from gym import register
    import gym

    register(id='halfcheetah-v2', entry_point=HalfCheetahEnv, max_episode_steps=1000)
    env = gym.make('halfcheetah-v2')

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

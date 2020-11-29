import numpy as np
import os

from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        # print("forward_reward: ", forward_reward, xposbefore, xposafter, self.dt)
        # contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            # reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def mb_step(self, obs, action, next_obs):
        forward_reward = (next_obs[0] - obs[0]) / self.dt
        ctrl_reward = .5 * np.square(action).sum()
        rewards = forward_reward - ctrl_reward + 1.
        notdone = np.isfinite(obs).all() and 0.2 <= obs[1] <= 1.0
        done = not notdone

        return rewards, done

    def _get_obs(self):
        return np.concatenate([
            self.get_body_com('torso').flat[:1],
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


if __name__ == '__main__':
    from gym import register
    import gym

    register(id='ant-v2', entry_point=AntEnv, max_episode_steps=1000)
    env = gym.make('ant-v2')

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

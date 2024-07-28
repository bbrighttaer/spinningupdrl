import copy

import numpy as np


class Episode:
    """
    Houses the experiences within a trajectory
    """

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []
        self.state = []
        self.next_state = []
        self.prev_action = []
        self.mask = []

    def add(self, obs, act, reward, next_obs, done, state, next_state, prev_action, mask=0):
        """
        Adds an episode record.

        Arguments
        ----------
        :param obs: observation_t
        :param act: action_t
        :param reward: reward_t
        :param next_obs: observation_{t+1}
        :param done: whether episode has ended
        :param state: state_t
        :param next_state: state_{t+1}
        :param prev_action: action_{t-1}
        :param mask: whether this record is a padding
        """
        self.obs.append(obs)
        self.action.append([act])
        self.reward.append([reward])
        self.next_obs.append(next_obs)
        self.done.append([done])
        self.state.append(state)
        self.next_state.append(next_state)
        self.prev_action.append([prev_action])
        self.mask.append([mask])

    def __len__(self):
        return len(self.obs)

    def pad_episode(self, size) -> "Episode":
        """
        Pads a copy of this episode by the given size.

        Arguments
        ----------
        :param size: padding size
        :return: Padded copy
        """
        ep_copy = copy.deepcopy(self)
        obs = ep_copy.obs[-1]
        state = ep_copy.state[-1]
        for i in range(size):
            ep_copy.add(
                obs=np.zeros_like(obs),
                act=0,
                reward=0,
                next_obs=np.zeros_like(obs),
                done=0,
                state=np.zeros_like(state),
                next_state=np.zeros_like(state),
                prev_action=0,
                mask=1
            )
        return ep_copy

    def deflate(self) -> 'Episode':
        episode = Episode()
        episode.add(
            obs=np.concatenate(self.obs),
            act=np.concatenate(self.action),
            reward=np.concatenate(self.reward),
            next_obs=np.concatenate(self.next_obs),
            done=np.concatenate(self.done),
            state=np.concatenate(self.state),
            next_state=np.concatenate(self.next_state),
            prev_action=np.concatenate(self.prev_action),
            mask=np.concatenate(self.mask),
        )
        return episode


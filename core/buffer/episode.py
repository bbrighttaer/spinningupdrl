import copy

import numpy as np

from core.proto.episode_proto import Episode as EpisodeProto


class Episode(EpisodeProto):
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

    def add(self, obs, act, reward, next_obs, done, state, next_state, prev_action, mask=False):
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
        def wrap(obj):
            return obj if isinstance(obj, np.ndarray) or isinstance(obj, list) else [obj]

        self.obs.append(wrap(obs))
        self.action.append(wrap(act))
        self.reward.append(wrap(reward))
        self.next_obs.append(wrap(next_obs))
        self.done.append(wrap(done))
        self.state.append(wrap(state))
        self.next_state.append(wrap(next_state))
        self.prev_action.append(wrap(prev_action))
        self.mask.append(wrap(mask))

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
                mask=True,
            )
        return ep_copy

    def merge_time_steps(self) -> 'Episode':
        """
        Merge all episode steps into a single into a single batch.

        :return: Episode with a single batch entry for each property.
        """
        episode = Episode()
        episode.add(
            obs=np.stack(self.obs),
            act=np.stack(self.action),
            reward=np.stack(self.reward),
            next_obs=np.stack(self.next_obs),
            done=np.stack(self.done),
            state=np.stack(self.state),
            next_state=np.stack(self.next_state),
            prev_action=np.stack(self.prev_action),
            mask=np.stack(self.mask),
        )
        return episode


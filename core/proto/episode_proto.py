from typing import Protocol


class Episode(Protocol):
    """
    Protocol for Episode classes
    """

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
        ...

    def __len__(self):
        ...

    def pad_episode(self, size) -> "Episode":
        """
        Pads a copy of this episode by the given size.

        Arguments
        ----------
        :param size: padding size
        :return: Padded copy
        """
        ...

    def merge_time_steps(self) -> 'Episode':
        """
        Merge all episode steps into a single into a single batch.

        :return: Episode with a single batch entry for each property.
        """
        ...
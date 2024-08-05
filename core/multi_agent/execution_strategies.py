import abc
import collections

from core import utils, constants


class RolloutWorkerExecutionStrategy(abc.ABC):
    """
    Parent class for different multi-agent execution strategies.
    A multi-agent execution strategy controls how the actions of agents within a time step (thinking step)
    are arrived at. One approach could be to use a hierarchical, synchronous decision-making approach, or different
    communication-based methods.
    """

    def __init__(self, rollout_worker):
        self.worker = rollout_worker
        self.policies = rollout_worker.policies
        self.policy_mapping_fn = rollout_worker.policy_mapping_fn

    @abc.abstractmethod
    def __call__(self, env, obs, state, action_mask, prev_actions, prev_hidden_states, prev_messages, explore, **kwargs):
        ...


class DefaultExecutionStrategy(RolloutWorkerExecutionStrategy):

    def __call__(self, env, obs, state, action_mask, prev_actions, prev_hidden_states, prev_messages, explore, **kwargs):
        # get the observation message of every policy
        messages = {}
        for policy_id in self.policies:
            policy = self.policies[policy_id]
            agent_id = self.policy_mapping_fn(policy_id)
            message = policy.get_message(
                obs=obs[agent_id],
                state=state[agent_id],
                action_mask=action_mask,
                prev_action=prev_actions[policy_id],
                prev_msg=prev_messages[policy_id],
            )
            messages[policy_id] = message

        # put together the received messages for each agent
        received_msgs = collections.defaultdict(list)
        for policy_i in self.policies:
            for policy_j, msg in messages.items():
                if policy_i != policy_j and msg is not None and len(msg) > 0:
                    received_msgs[policy_i].append(msg)

        # action selection
        actions_dict = {}
        hidden_states = {}
        for policy_id in self.policies:
            policy = self.policies[policy_id]
            agent_id = self.policy_mapping_fn(policy_id)
            action, hidden_state = policy.compute_action(
                obs=obs[agent_id],
                prev_action=prev_actions[policy_id],
                prev_hidden_state=prev_hidden_states[policy_id],
                explore=explore,
                shared_messages=received_msgs[policy_id],
                state=state[agent_id],
                action_mask=action_mask[agent_id],
            )
            actions_dict[agent_id] = action
            hidden_states[policy_id] = hidden_state

        # send selected actions to environment
        next_obs, reward, done, truncated, info = env.step(actions_dict)
        dones = {}
        for policy_id in self.policies.keys():
            agent_id = self.policy_mapping_fn(policy_id)
            if "__all__" in done:
                dones[policy_id] = done["__all__"]
            else:
                dones[policy_id] = done[agent_id]

        return {
            constants.OBS: obs,
            constants.STATE: state,
            constants.ACTION_MASK: action_mask,
            constants.REWARD: reward,
            constants.NEXT_OBS: self.worker.unpack_env_data(next_obs, constants.OBS),
            constants.NEXT_STATE: self.worker.unpack_env_data(next_obs, constants.STATE),
            constants.NEXT_ACTION_MASK: self.worker.unpack_env_data(next_obs, constants.ACTION_MASK),
            constants.DONE: dones,
            constants.INFO: info,
            constants.HIDDEN_STATE: hidden_states,
            constants.ACTION: {
                self.policy_mapping_fn(agent, reverse=True): actions_dict[agent] for agent in actions_dict
            },
            constants.SENT_MESSAGE: messages,
            constants.RECEIVED_MESSAGE: received_msgs
        }


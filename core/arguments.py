import argparse

from core import constants
from core.utils import DotDic


def cmd_arguments():
    parser = argparse.ArgumentParser("Command line arguments for experiments")

    parser.add_argument(
        "--seed",
        type=int,
        default=321,
    )

    parser.add_argument(
        "--env_id",
        type=str,
        default="CartPole-v1",
        help="The environment to use for the experiment. It should be an already registered Gym API compatible env."
    )

    parser.add_argument(
        "--disable_torch_deterministic",
        action="store_true",
        help="Sets the value of torch.backends.cudnn.deterministic`"
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="sample",
        help="The name of the running algorithm",
    )

    parser.add_argument(
        "--mode",
        default="sa",
        choices=[constants.SINGLE_AGENT, constants.MULTI_AGENT, constants.MULTI_AGENT_WITH_PARAMETER_SHARING],
        help="Experiment mode. One of single-agent (sa), multi-agent (ma), multi-agent with parameter sharing (ma-ps)."
    )

    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1000000,
        help="Total number of time steps of the experiment."
    )

    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=10000,
        help="Number of training time steps before an evaluation phase."
    )

    parser.add_argument(
        "--evaluation_num_episodes",
        type=int,
        default=20,
        help="The number of episodes to run per each evaluation phase."
    )

    args = parser.parse_args()
    args = DotDic(args.__dict__)

    return args

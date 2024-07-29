import importlib
import json
import logging
import os
import time

import torch.cuda
from torch.utils.tensorboard import SummaryWriter

from core import constants
from core import utils
from core import single_agent
from core.simple_callback import SimpleCallback
from core.logging import Logger


class Runner:
    """
    Runs an instance of an experiment.

    Arguments
    ----------
    args - Command line arguments
    """

    def __init__(self, args):
        self.cmd_args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()

        # get running algorithm
        algo = self.cmd_args.algo

        # get default configs of the algorithm and update using cmd args
        default_configs = importlib.import_module(f"config.{algo}_config")
        running_config = default_configs.RUNNING_CONFIG
        running_config["device"] = self.device
        running_config = utils.DotDic(utils.update_dict(running_config, self.cmd_args.__dict__))
        algo_config = utils.DotDic(utils.update_dict(default_configs.ALGO_CONFIG, self.cmd_args.__dict__))
        model_config = utils.DotDic(utils.update_dict(default_configs.MODEL_CONFIG, self.cmd_args.__dict__))
        env_config = utils.DotDic(utils.update_dict(default_configs.ENV_CONFIG, self.cmd_args.__dict__))
        config = {
            constants.RUNNING_CONFIG: running_config,
            constants.ALGO_CONFIG: algo_config,
            constants.MODEL_CONFIG: model_config,
            constants.ENV_CONFIG: env_config,
        }

        # create experiment directory for all items that would be saved to file
        running_name = f"{algo}_{self.cmd_args.mode}_{utils.generate_random_label()}"
        working_dir = os.path.join("exp_results", running_name)
        os.makedirs(working_dir, exist_ok=True)

        # create loggers
        summary_writer = SummaryWriter(log_dir=working_dir + "/runs/")
        logger = Logger(
            exp_name=running_name,
            working_dir=working_dir,
            level=logging.DEBUG,
        )
        logger.debug(f"Running experiment {running_name}, working dir={working_dir}, seed={self.cmd_args.seed}")

        # save config to file
        with open(working_dir + "/config.json", "w") as f:
            json.dump(config, f)

        # callback class
        exp_callback = SimpleCallback(summary_writer, logger)

        # run experiment based on mode
        if self.cmd_args.mode == constants.SINGLE_AGENT:
            # create policy(ies) using PolicyCreator
            policy, replay_buffer = single_agent.SingleAgentPolicyCreator(config, summary_writer, logger)

            # create rollout worker
            rollout_worker = single_agent.RolloutWorkerCreator(
                policy, replay_buffer, config, summary_writer, logger, callback=exp_callback
            )

            # create training worker
            training_worker = single_agent.SingleAgentTrainingWorker(
                policy, replay_buffer, config, summary_writer, logger, callback=exp_callback
            )

            # training loop
            while rollout_worker.get_global_time_step() < self.cmd_args.total_time_steps:
                # check for evaluation step
                time_step = rollout_worker.get_global_time_step()
                if time_step > 0 and time_step % self.cmd_args.evaluation_interval == 0:
                    rollout_worker.evaluate_policy(self.cmd_args.evaluation_num_episodes)

                # generate an episode
                rollout_worker.generate_trajectory()

                # training
                training_worker.train(time_step)

        elif self.cmd_args.mode == constants.MULTI_AGENT:
            ...

        elif self.cmd_args.mode == constants.MULTI_AGENT_WITH_PARAMETER_SHARING:
            ...

        logger.info(f"Total time elapsed is {time.perf_counter() - start_time} seconds")
        summary_writer.close()

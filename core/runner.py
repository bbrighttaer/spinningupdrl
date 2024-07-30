import importlib
import json
import logging
import os
import time

import torch.cuda
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

from core import constants
from core import utils
from core import single_agent
from core.metrics.sim_metrics import MetricsManager
from core.simple_callback import SimpleCallback
from core.logging import Logger

# random code for the experiment
TRIAL_CODE = utils.generate_random_label()


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

    def start(self):
        start_time = time.perf_counter()

        # get running algorithm
        algo = self.cmd_args.algo

        # get default configs of the algorithm and update using cmd args
        default_configs = importlib.import_module(f"config.{algo}_config")
        running_config = default_configs.RUNNING_CONFIG
        running_config["device"] = self.device
        running_config = utils.DotDic(utils.update_dict(running_config, self.cmd_args))
        algo_config = utils.update_dict(default_configs.ALGO_CONFIG, self.cmd_args)
        algo_config["algo"] = algo
        algo_config = utils.DotDic(algo_config)
        model_config = utils.DotDic(utils.update_dict(default_configs.MODEL_CONFIG, self.cmd_args))
        env_config = utils.DotDic(utils.update_dict(default_configs.ENV_CONFIG, self.cmd_args))
        config = {
            constants.RUNNING_CONFIG: running_config,
            constants.ALGO_CONFIG: algo_config,
            constants.MODEL_CONFIG: model_config,
            constants.ENV_CONFIG: env_config,
            constants.CMD_LINE_ARGS: self.cmd_args,
        }

        # create experiment directory for all items that would be saved to file
        trial_name = f"{algo}_{self.cmd_args.mode}_{TRIAL_CODE}"
        running_config[constants.TRIAL_NAME] = trial_name
        working_dir = os.path.join("exp_results", trial_name)
        os.makedirs(working_dir, exist_ok=True)

        # create loggers
        summary_writer = SummaryWriter(log_dir=working_dir + "/runs/")
        logger = Logger(
            trial_name=trial_name,
            working_dir=working_dir,
            level=logging.DEBUG,
        )

        # print experiment info
        exp_info = PrettyTable(field_names=["trial name", "seed", "device", "total_timesteps", "working dir"])
        exp_info.add_row([trial_name, self.cmd_args.seed, self.device, running_config.total_timesteps, working_dir])
        logger.info(str(exp_info))

        # save config to file
        with open(working_dir + "/config.json", "w") as f:
            json.dump(config, f)

        # callback class
        exp_callback = SimpleCallback(summary_writer, logger)

        # metrics handling
        metrics_manager = MetricsManager(config, working_dir, logger)

        # run experiment based on mode
        if self.cmd_args.mode == constants.SINGLE_AGENT:
            # create policy(ies) using PolicyCreator
            policy, replay_buffer = single_agent.SingleAgentPolicyCreator(config, summary_writer, logger)

            # create rollout worker
            rollout_worker = single_agent.RolloutWorkerCreator(
                policy, replay_buffer, config, summary_writer, metrics_manager, logger, callback=exp_callback
            )

            # create training worker
            training_worker = single_agent.SingleAgentTrainingWorker(
                policy, replay_buffer, config, summary_writer, metrics_manager, logger, callback=exp_callback
            )

            # training loop
            last_eval_step = 0
            while rollout_worker.timestep < self.cmd_args.total_timesteps:
                # check for evaluation step
                timestep = rollout_worker.timestep
                if timestep > (self.cmd_args.evaluation_interval + last_eval_step):
                    rollout_worker.evaluate_policy(self.cmd_args.evaluation_num_episodes)
                    last_eval_step = timestep

                # generate an episode
                rollout_worker.generate_trajectory()

                # training
                training_worker.train(timestep, rollout_worker.cur_iter)

            # completion protocol
            rollout_worker.evaluate_policy(self.cmd_args.evaluation_num_episodes)

        elif self.cmd_args.mode == constants.MULTI_AGENT:
            ...

        elif self.cmd_args.mode == constants.MULTI_AGENT_WITH_PARAMETER_SHARING:
            ...

        logger.info(f"Total time elapsed is {time.perf_counter() - start_time} seconds")
        summary_writer.close()

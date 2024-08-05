import typing
from queue import Queue

from algos import Policy
from core import constants, utils
from core.envs import SMACv0
from core.metrics.sim_metrics import MetricsManager
from core.proto.callback_proto import Callback
from core.proto.rollout_worker_proto import RolloutWorker
from core.utils import DotDic


class DefaultCallback(Callback):
    """
    Implements callback functions for performing activities at different points in the training.
    """

    def __init__(self, metrics_manager: MetricsManager, logger):
        self.metrics_manager = metrics_manager
        self.logger = logger

        # For win-rate stats on SMAC
        self._smac_stats_queues = self._create_smac_stats_queues()

    def _create_smac_stats_queues(self):
        def create_queues():
            return DotDic({
                "battle_win_queue": Queue(maxsize=100),
                "ally_survive_queue": Queue(maxsize=100),
                "enemy_killing_queue": Queue(maxsize=100),
            })
        return {
            constants.TRAINING: create_queues(),
            constants.EVALUATION: create_queues(),
        }

    def on_episode_end(
            self, worker: RolloutWorker,
            episode_stats: typing.Dict[str, constants.Number],
            is_training: bool,
            **kwargs,
    ):
        # record SMAC stats if using SMAC
        env = worker.env.unwrapped  # get environment
        if isinstance(env, SMACv0):
            queues_key = constants.TRAINING if is_training else constants.EVALUATION
            queues = self._smac_stats_queues[queues_key]
            smac_stats = utils.get_smac_stats(
                death_tracker_ally=env.death_tracker_ally,
                death_tracker_enemy=env.death_tracker_enemy,
                battle_win_queue=queues.battle_win_queue,
                ally_survive_queue=queues.ally_survive_queue,
                enemy_killing_queue=queues.enemy_killing_queue,
            )
            episode_stats.update(smac_stats)

        # log metrics using metrics manager
        self._log_metrics(
            cur_iter=worker.cur_iter,
            timestep=worker.timestep,
            mode=constants.TRAINING if is_training else constants.EVALUATION,
            exp_stats=episode_stats,
        )

    def _log_metrics(self, cur_iter, timestep, mode, exp_stats):
        self.metrics_manager.add_performance_metric(
            data=exp_stats,
            cur_iter=cur_iter,
            timestep=timestep,
            training=mode == constants.TRAINING,
        )

    def on_learn_on_batch_end(
            self, policy: Policy,
            cur_iter: int,
            timestep: int,
            learning_stats: typing.Dict[str, constants.Number],
            **kwargs,
    ):
        # update metrics
        self.metrics_manager.add_learning_stats(
            cur_iter=cur_iter,
            timestep=timestep,
            data=learning_stats,
            label_suffix=kwargs.get("label_suffix")
        )

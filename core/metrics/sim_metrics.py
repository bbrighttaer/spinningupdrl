import time

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from core import metrics, constants


def _generate_cols_for_metric(m):
    return [m, f"{m}_mean", f"{m}_max", f"{m}_min"]


def _parse_data_dict(data: dict):
    return {
        str(k): data[k] for k in data
    }


class MetricsManager:
    """
    Helper class to manage simulation metrics
    """

    def __init__(self, config, working_dir, summary_writer, logger):
        self._start_time = time.perf_counter()
        self.config = config
        self.base_dir = working_dir
        self.summary_writer = summary_writer
        self.logger = logger
        self._window_size = 100
        self._last_logging_step = 0

        # initialize tables
        self.common_cols = [
            constants.TRIAL_NAME,
            constants.TIMESTEP,
            constants.ITER,
            constants.TOTAL_TIME,
        ]
        self._learning_stats_df = pd.DataFrame({
            str(k): [] for k in self.common_cols + list(metrics.LearningMetrics)
        })
        perf_metrics = []
        for m in list(metrics.PerformanceMetrics):
            perf_metrics.extend(_generate_cols_for_metric(m))
        df_columns = self.common_cols + perf_metrics
        self._training_stats_df = pd.DataFrame({
            str(k): [] for k in df_columns
        })
        self._evaluation_stats_df = pd.DataFrame({
            str(k): [] for k in df_columns
        })

    def add_learning_stats(self, cur_iter: int, timestep: int, data: dict, label_suffix: str = None):
        elapsed_time = time.perf_counter() - self._start_time
        running_config = self.config[constants.RUNNING_CONFIG]
        data = _parse_data_dict(data)

        # ensure new metrics are added
        new_metrics = set(data.keys()) - set(self._learning_stats_df.columns)
        for m in new_metrics:
            self._learning_stats_df[m] = [np.nan] * len(self._learning_stats_df)

        # construct row data
        row_data = {
            constants.TRIAL_NAME: [running_config[constants.TRIAL_NAME]],
            constants.TIMESTEP: [timestep],
            constants.ITER: [cur_iter],
            constants.TOTAL_TIME: [elapsed_time],
        }
        row_data.update({k: [v] for k, v in data.items()})

        # logging
        label = f"{constants.TRAINING}/{constants.LEARNING}"
        if label_suffix:
            label += f"/{label_suffix}"
        self._log_scalars_to_tensorboard(label, row_data, timestep)

        # append row to data table
        df2 = pd.DataFrame(row_data)
        self._learning_stats_df = pd.concat([self._learning_stats_df, df2], ignore_index=True)

        # save to disk
        filename = constants.LEARNING
        if label_suffix:
            filename += f"_{label_suffix}"
        self._learning_stats_df.to_csv(f"{self.base_dir}/{filename}_stats.csv", index=False)

    def add_performance_metric(self, cur_iter: int, timestep: int, data: dict, training: bool):
        elapsed_time = time.perf_counter() - self._start_time
        running_config = self.config[constants.RUNNING_CONFIG]
        data = _parse_data_dict(data)

        # get active data table
        if training:
            dataframe = self._training_stats_df
            prefix = constants.TRAINING
        else:
            dataframe = self._evaluation_stats_df
            prefix = constants.EVALUATION

        # ensure new metrics are added
        new_metrics = set(data.keys()) - set(dataframe.columns.values.tolist())
        for m in new_metrics:
            new_cols = _generate_cols_for_metric(m)
            for col in new_cols:
                dataframe[col] = [np.nan] * len(dataframe)

        # construct row data
        row_data = {
            constants.TRIAL_NAME: [running_config[constants.TRIAL_NAME]],
            constants.TIMESTEP: [timestep],
            constants.ITER: [cur_iter],
            constants.TOTAL_TIME: [int(elapsed_time)],
        }
        for key, value in data.items():
            past_vals = dataframe[key][-(self._window_size - 1):].values
            window_vals = np.append(past_vals, value)

            # new row in records for this key
            key = str(key)
            row_data[key] = [value]
            row_data[f"{key}_mean"] = [np.mean(window_vals)]
            row_data[f"{key}_max"] = [np.max(window_vals)]
            row_data[f"{key}_min"] = [np.min(window_vals)]

        # tb, console, and file logging
        self._log_scalars_to_tensorboard(prefix, row_data, timestep)
        if timestep >= (running_config.logging_steps + self._last_logging_step):
            field_names = []
            field_values = []
            for col in row_data:
                if constants.EVALUATION not in col:
                    field_names.append(col)
                    field_values.append(row_data[col][0])

            info = PrettyTable(field_names=field_names)
            info.add_row(field_values)
            self.logger.info(str(info))
            if training:
                self._last_logging_step = timestep

        # append row to data table and save to disk
        df2 = pd.DataFrame(row_data)
        if training:
            is_empty = len(self._training_stats_df) == 0
            self._training_stats_df = df2 if is_empty else pd.concat([dataframe, df2], ignore_index=True)
            self._training_stats_df.to_csv(
                f"{self.base_dir}/{prefix}_perf_stats.csv", index=False
            )
        else:
            self._evaluation_stats_df = pd.concat([dataframe, df2], ignore_index=True)
            self._evaluation_stats_df.to_csv(
                f"{self.base_dir}/{prefix}_perf_stats.csv", index=False
            )

    def _log_scalars_to_tensorboard(self, prefix, row_data, timestep):
        # tensorboard logging
        for key in row_data:
            if key not in self.common_cols:
                # tb logging
                self.summary_writer.add_scalar(
                    tag=f"{prefix}/{key}",
                    scalar_value=row_data[key][0],
                    global_step=timestep,
                )

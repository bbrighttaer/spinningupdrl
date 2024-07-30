import random

import numpy as np
import torch

from core import arguments, runner, utils

if __name__ == "__main__":
    # parse command line arguments, if any
    args = arguments.cmd_arguments()

    # random code for the experiment
    trial_code = utils.generate_random_label()

    # seeding for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = not args.disable_torch_deterministic

    # create experiment runner
    exp_runner = runner.Runner(trial_code, args)

    # begin experiment
    results = exp_runner()

    # persist final results


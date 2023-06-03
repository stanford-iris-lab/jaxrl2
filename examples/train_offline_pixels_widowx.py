#! /usr/bin/env python
import os

os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
from jaxrl2.agents.cql_encodersep.pixel_cql_learner import PixelCQLLearnerEncoderSep
from jaxrl2.wrappers.dummy_env import DummyEnv
from jaxrl2.data.eps_transition_dataset import EpisodicTransitionDataset
from jaxrl2.utils.general_utils import add_batch_dim
import collections

import numpy as np

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from jaxrl2.data.widowx_configs import *

from examples.train_utils_sim import (
    offline_training_loop,
)


def main(variant):
    import jax

    variant.stochastic_evals = False

    kwargs = variant["train_kwargs"]
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = variant.max_steps

    if variant.suffix:
        expname = (
            create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
        )
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
    outputdir = os.environ["EXP"] + "/jaxrl/" + expname
    variant.outputdir = outputdir
    print("writing to output dir ", outputdir)

    group_name = variant.prefix + "_" + variant.launch_group_id
    wandb_logger = WandBLogger(
        variant.prefix != "",
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=outputdir,
        group_name=group_name,
    )

    env = DummyEnv()

    sample_obs = add_batch_dim(env.observation_space.sample())
    sample_action = add_batch_dim(env.action_space.sample())
    print("sample obs shapes", [(k, v.shape) for k, v in sample_obs.items()])
    print("sample action shapes", sample_action.shape)

    agent = PixelCQLLearnerEncoderSep(variant.seed, sample_obs, sample_action, **kwargs)

    if variant.restore_path != "":
        agent.restore_checkpoint(
            variant.restore_path,
            reset_critic=variant.reset_critic,
            rescale_critic_last_layer_ratio=variant.rescale_critic_last_layer_ratio,
        )

    online_replay_buffer = None
    if not variant.online_from_scratch:
        if variant.rew_func_for_target_only:
            curr_rft = (
                variant.reward_func_type if hasattr(variant, "reward_func_type") else 0
            )
            variant.reward_func_type = 0

        if variant.rew_func_for_target_only:
            variant.reward_func_type = curr_rft

        if variant.get("online_bound_nstep_return", -1) > 0:
            print("setting nstep return off during offline phase")
            agent.online_bound_nstep_return = -1
        
        config_type = variant.get('dataset', 'debug')
        if config_type == 'debug':
            dataset_paths = debug_config()
        elif config_type == 'sorting':
            dataset_paths = sorting_dataset()
        elif config_type == 'pickplace':
            dataset_paths = pickplace_dataset()
        elif config_type == 'sorting_pickplace':
            dataset_paths = sorting_pickplace_dataset()
        else:
            raise ValueError(f"Unknown dataset type {config_type}")

        replay_buffer = EpisodicTransitionDataset(dataset_paths)

        offline_training_loop(
            variant,
            agent,
            env,
            replay_buffer,
            None,
            wandb_logger,
            perform_control_evals=False,
        )
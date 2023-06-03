# copy this into wandb_config.py!
WANDB_API_KEY='a63af51a017582ae928e7cfd56bd21d57bb45def'
WANDB_EMAIL='asap7772@berkeley.edu'
WANDB_USERNAME='asap7772'

def get_wandb_config():
    return dict(
        WANDB_API_KEY=WANDB_API_KEY,
        WANDB_EMAIL=WANDB_EMAIL,
        WANDB_USERNAME=WANDB_USERNAME,
    )
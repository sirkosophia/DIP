import os
import torch
import wandb
import random
import argparse
import numpy as np

from hummingbird.src.hbird_eval import hbird_evaluation

import yaml
import torch.nn.functional as F
from models.humming_net import Humming_net
 

def load_model_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config

def main(args, config):
    print(f"the script arguments are {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Humming_net(**config)
    
    embedding_size = config['embedding_size']
    print(f"Embedding size: {embedding_size}")
    def token_features(model, imgs):
        return model(imgs), None

    hbird_miou = hbird_evaluation(
        model.to(device),
        d_model=embedding_size,
        patch_size=args.patch_size,
        batch_size = args.batch_size,
        input_size=args.input_size,
        augmentation_epoch=args.augmentation_epoch,
        device=device,
        num_neighbour=30,
        nn_params=None,
        ftr_extr_fn=token_features,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        memory_size=args.memory_size,
        beta=args.beta,        
    )
    

    print(f"Hummingbird Evaluation (mIoU): {hbird_miou}")

    with open('eval_scores.txt', 'a') as file:
        file.write( f"{str(args.data_dir)}, {str(args.name)}, {str(config['model_params'])}, {str(config['model_weights'])}: {hbird_miou}\n")

    print("Score written to file.")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # wandb.login()

    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    # Standard arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")

    # Model arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")
    parser.add_argument("--patch-size", type=int, default=16, help="Size of the model patches")
    parser.add_argument("--memory-size", type=int, default=None, help="The size of the memory bank. Unbounded if not specified")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="VOCSegmentation", help="Path to dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--name", type=str, required=True, help="Name of experiment")
    parser.add_argument("--augmentation_epoch", type=int, default=2, help="number of augmentation epochs")
    parser.add_argument("--beta", type=float, default=0.02, help="temperature for the cross attention")
    parser.add_argument("--dataset_name", type=str, default="voc", help="name of dataset")
    
    args = parser.parse_args()
    print(f"Args: {args}")
    config = load_model_config(args.config)

    seed_everything(args.seed)
    main(args, config)

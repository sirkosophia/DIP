
import yaml
import subprocess
import os
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(prog="PROG")
    
    
    parser.add_argument("--name", "-n", type=str, default="dip", required=False)
    parser.add_argument("--batch_size", "-bs", type=int, default=32, required=False)
    parser.add_argument("--image_backbone", "-ib", type=str, default="small",required=False)
    parser.add_argument("--dataset", "-ds", type=str, default="VOCSegmentation",required=False)
    parser.add_argument("--input_size", "-is", type=int, default=224, required=False)
    parser.add_argument("--additional_name", "-an", type=str, default=None, required=False)
    parser.add_argument("--model_weights", "-mw", type=str, default=None, required=False)
    parser.add_argument("--dataset_name", "-dn", type=str, default="voc", required=False)
    parser.add_argument("--augmentation_epochs", "-ae", type=int, default=1, required=False)
    parser.add_argument("--beta", "-beta", type=float, default=0.07, required=False)
    parser.add_argument("--memory_size", "-ms", type=int, default=102400, required=False)
    parser.add_argument("--mlp_ratio", "-mlpr", type=int, default=4, required=False)
    parser.add_argument("--mlp_out_features", "-mlpout", type=int, default=384, required=False)


    args = parser.parse_args()
    backbone = 'dip'
    print("args.name", args.name)
    # -----  BEGIN -----
    embed_dim = 384
    if args.image_backbone == "base":
        backbone = "dip_b"
        embed_dim =768 
    elif args.image_backbone=="large":
        backbone = "dip_l"
        embed_dim = 1024
    args.model_name = "DIP"
    
    print("args.image_backbone", args.image_backbone)
    print("embed_dim", embed_dim)
        

    name = backbone  + "_input_size" + str(args.input_size) + "_image_backbone_" + args.image_backbone
    if args.additional_name is not None:
        name = name + "_" + args.additional_name    

    if args.mlp_out_features != 384:
        name = name + "_mlpo" + str(args.mlp_out_features)

    if args.dataset_name == "ade20k":
        args.dataset = "data/ADEChallengeData2016"
    else: 
        dir_location = "data/"
        if os.path.exists(dir_location):
            args.dataset = dir_location + args.dataset
        
        
    name = name + "_beta" + str(args.beta)
    name = name + "_augmentation_epochs" + str(args.augmentation_epochs)

    name = name + "_dataset_name" + str(args.dataset_name)
        

    
    print("RUN NAME: ", name )
    # Define the initial structure of the YAML file

    patch_size=14
    num_patches = (args.input_size // patch_size)**2

    print("args.dataset", args.dataset)
    print("args.dataset_name", args.dataset_name)
    data_template = {
                'model_name': args.model_name,
                'embedding_size': embed_dim,
                'model_params': {
                    "enc_embed_dim": embed_dim,
                    'image_backbone': args.image_backbone,
                    "mlp_ratio": args.mlp_ratio,
                    "mlp_out_features": args.mlp_out_features,
                },
                'model_weights': args.model_weights,
            }


    yaml_filename = f'hummingbird/configs_dip_hummingbird/eval_humm_{backbone}_{name}.yaml'
 
    with open(yaml_filename, 'w') as file:
        yaml.dump(data_template, file, default_flow_style=False, sort_keys=False)

    cmd = f"python humm_eval.py                  \
        --seed=42                   \
        --batch-size={args.batch_size}            \
        --input-size={args.input_size}            \
        --patch-size={patch_size}             \
        --memory-size={args.memory_size}        \
        --augmentation_epoch={args.augmentation_epochs}  \
        --data-dir={args.dataset}  \
        --name={name}  \
        --config={yaml_filename} \
        --beta={args.beta} \
        --dataset_name={args.dataset_name}  "
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()

import torch
import numpy as np
from TCN.word_cnn.model import TCN
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

import torch
import numpy as np
import argparse
import yaml
from TCN.word_cnn.model import TCN
import datautils

def all_retrieval(model, num, config):
    L=config["retrieval"]["L"]
    H=config["retrieval"]["H"]
    train_set = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"],flag='train',size=[L, 0, L])
    all_repr=torch.load(config["path"]["vec_path"])
    references=[]
    with torch.no_grad():
        for i in range(len(train_set) - L - H + 1):
            x=train_set.data_x[i:i+L]
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).transpose(1, 2).to(config["retrieval"]["device"])
            x_vec = model.encode(x)
            k=x_vec.shape[-2]
            l=x_vec.shape[-1]
            x_vec=x_vec.reshape(1,k*l)
            all_repr = all_repr.reshape(config["retrieval"]["length"],k*l)
            distances=torch.norm(x_vec.cpu() - all_repr,dim=1)
            _, idx=torch.topk(-1*distances, num)
            references.append(idx.int())
        references = torch.cat(references, dim=0)
        torch.save(references, config["path"]["ref_path"])
    return references

def all_encode(model,config):
    hisvec_list=[]
    reference_list=[]
    L=config["retrieval"]["L"]
    H=config["retrieval"]["H"]
    print(f"L={L}, H={H}")
    train_set = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"],flag='train',size=[L, 0, L])
    print(f"train_set length: {len(train_set)}")
    print(f"data_x shape: {train_set.data_x.shape}")
    total_steps = len(train_set) - L - H + 1
    print(f"Total steps: {total_steps}")
    with torch.no_grad():
        for i in range(len(train_set) - L - H + 1):
            if i % 10 == 0:
                print(f"Processing step {i}/{total_steps}")
            x=train_set.data_x[i:i+L]
            y=train_set.data_x[i+L:i+L+H]
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).transpose(1, 2).to(config["retrieval"]["device"])
            x_vec = model.encode(x)
            hisvec_list.append(x_vec.cpu())
    print("Concatenating results...")
    hisvec_list = torch.cat(hisvec_list, dim=0)
    print(f"Final shape: {hisvec_list.shape}")
    print(f"Saving to {config['path']['vec_path']}...")
    torch.save(hisvec_list.float(), config["path"]["vec_path"])
    print("Encoding completed!")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="retrieval_ele.yaml")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--type", type=str, default="encode", choices=["encode", "retrieval"]
    )
    parser.add_argument("--encoder", default="TCN")

    args = parser.parse_args()
    print(args)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["retrieval"]["encoder"] = args.encoder
    model = TCN(
            input_size=config["retrieval"]["length"],
            output_size=config["retrieval"]["length"], num_channels=[config["retrieval"]["length"]] * (config["retrieval"]["level"]) + [config["retrieval"]["length"]],
        ).to(config["retrieval"]["device"])
    model=torch.load(config["path"]["encoder_path"], weights_only=False, map_location=config["retrieval"]["device"])
    if args.type == 'encode':
        all_encode(model,config)
    if args.type == 'retrieval':
        all_retrieval(model, 10, config)


    

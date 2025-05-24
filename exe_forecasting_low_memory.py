import argparse
import torch
import datetime
import json
import yaml
import os
import gc

from main_model import RATD_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate

# 设置PyTorch内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description="RATD")
parser.add_argument("--config", type=str, default="low_memory_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=50)  # 从100减少到50
parser.add_argument("--target_dim", type=int, default=161)  # 从321减少到161
parser.add_argument("--h_size", type=int, default=168)
parser.add_argument("--ref_size", type=int, default=96)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["diffusion"]["h_size"] = args.h_size
config["diffusion"]["ref_size"] = args.ref_size
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# 清理内存
torch.cuda.empty_cache()
gc.collect()

train_loader, valid_loader, test_loader = get_dataloader(
    device= args.device,
    batch_size=config["train"]["batch_size"],
)

model = RATD_Forecasting(config, args.device, args.target_dim).to(args.device)

# 清理内存
torch.cuda.empty_cache()
gc.collect()

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("save/" + args.modelfolder + "/model.pth"))

# 清理内存
torch.cuda.empty_cache()
gc.collect()

model.target_dim = args.target_dim
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=1,
    mean_scaler=0,
    foldername=foldername,
) 
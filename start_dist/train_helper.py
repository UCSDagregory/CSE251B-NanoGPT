import json
import model
import os

from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
import torch
import inspect

from train_helper_loader import BatchHelper

param_file_name = "training_model_params.json"
METADATA_KEYS = ["author"]

def CreateModel(model_folder_name:str, checkpoint_file_path:str=None, chkpt_folder_name_init:str=None, from_scratch=True) -> model.nanoGPT:
    param_file_path = os.path.join(os.getcwd(), model_folder_name, param_file_name)
    # Read from train
    if (from_scratch):
        args = [model_folder_name, chkpt_folder_name_init]
        with open(param_file_path, 'r') as file:
            data = json.load(file)
            # Metadata args
            for idx in range(len(METADATA_KEYS)):
                try:
                    args.append(data[METADATA_KEYS[idx]])
                except:
                    args.append("N/A")
            
            # Model param args
            for key in data:
                if (key in METADATA_KEYS):
                    continue
                args.append(data[key])

        gpt_model = model.nanoGPT(*args)
        return gpt_model
    else:
        try:
            if (checkpoint_file_path == None or not checkpoint_file_path.isalnum):
                raise ValueError("Checkpoint name cannot be none if not creating from scratch.\n")
        except:
                raise ValueError("Checkpoint name cannot be none if not creating from scratch.\n")
        gpt_model, model_sd, opt_args, opt_sd, iter_num, other_args = model.loadFromCheckpoint(model_folder_name, checkpoint_file_path, BatchHelper)
        return gpt_model, model_sd, opt_args, opt_sd, iter_num, other_args
    
def CreateOptimizer(args=None):
    if (args is None):
        raise ValueError("Invalid args passed to CreateOptimizer")
    param_groups, type, device_type = args

    if (type == 'muon'):
        use_distributed_muon = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if use_distributed_muon:
            print("Using distributed MuonWithAuxAdam")
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            print("Using SingleDeviceMuonWithAuxAdam")
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        return optimizer
    
    if (type == 'adam'):
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        learning_rate, betas = param_groups[-2:]
        param_groups = param_groups[:-2]
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
                param_groups,
                lr=learning_rate,
                betas=betas,
                **extra_args,
            )
        return optimizer
    
    raise ValueError("Invalid optimizer type.")
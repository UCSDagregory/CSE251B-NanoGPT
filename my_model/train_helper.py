import json
import model
import os

param_file_name = "training_model_params.json"
METADATA_KEYS = ["author"]

def createModel(model_folder_name:str, checkpoint_file_path:str=None, chkpt_folder_name_init:str=None, from_scratch=True) -> model.nanoGPT:
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
                args.append(int(data[key]))

        gpt_model = model.nanoGPT(*args)
        return gpt_model
    else:
        try:
            if (checkpoint_file_path == None or not checkpoint_file_path.isalnum):
                raise ValueError("Checkpoint name cannot be none if not creating from scratch.\n")
        except:
                raise ValueError("Checkpoint name cannot be none if not creating from scratch.\n")
        gpt_model, model_sd, opt_args, opt_sd = model.loadFromCheckpoint(model_folder_name, checkpoint_file_path)
        return gpt_model, model_sd, opt_args, opt_sd
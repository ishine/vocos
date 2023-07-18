import torch
model_path = "logs/lightning_logs/version_26/checkpoints/last.ckpt"
ckpt = torch.load(model_path, map_location="cpu")
state_dict_ = ckpt['state_dict']
to_pop_ks = [i for i in state_dict_ if "discriminators" in i or 'melspec_loss' in i]
[state_dict_.pop(i) for i in to_pop_ks]
print(state_dict_.keys())

torch.save(state_dict_, "pytorch_model.bin")

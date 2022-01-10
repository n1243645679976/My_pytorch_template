import torch
import os
from utils.dynamic_import import dynamic_import

def get_model(model, model_conf, resume, device):
    model_class = dynamic_import(model)
    model = model_class(model_conf)
    if resume:
        model.load_state_dict(resume)
    model.to(device)
    model.train()
    return model

def save_model(trainer):
    os.makedirs(os.path.join(trainer.args.exp, 'result'), exist_ok=True)
    model = {
        'iters': trainer.iters,
        'args': trainer.args,
        'conf': trainer.conf,
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict()
    }
    torch.save(model, os.path.join(trainer.exp, 'result', f'snapshot.{trainer.iters}'))

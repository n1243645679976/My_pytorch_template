import torch
from torch.optim import Adam
from utils.dynamic_import import dynamic_import
def get_optimizer(model, conf, load_optimizer):
    if 'type' in conf:
        optim_class = dynamic_import(conf['type'])
        optim = optim_class(model.parameters(), **conf['conf'])
        print(f'set optimizer {conf["type"]}')
    else: # default: Adam
        optim = Adam(model.parameters(), **conf['conf'])
        print(f'not setting optimizer, using Adam')
    
    optim.zero_grad()
    if load_optimizer:
        optim.load_state_dict(load_optimizer)
    
    return optim

def get_scheduler(optimizer, conf):
    if 'type' in conf:
        scheduler_class = dynamic_import(conf['type'])
        scheduler = scheduler_class(optimizer, **conf['conf'])
        print(f'set scheduler {conf["type"]}')
    else: # default: Adam
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.9,
            patience=0,
            verbose=False,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=1,
            min_lr=1e-4,
            eps=1e-08,
        )
        print(f'not setting scheduler, using ReduceLROnPlateau')
    return scheduler

if __name__ == '__main__':
    from utils.parse_config import get_train_config
    import torch
    args, conf = get_train_config()
    model = torch.nn.Sequential(torch.nn.Linear(100,100), torch.nn.ReLU(),
                                torch.nn.Linear(100,1))
    adam = get_optimizer(model, conf['optimizer'], load_optimizer=False)
    torch.save(adam.state_dict(), '.test_optim', _use_new_zipfile_serialization=False)
    
    load_optim = torch.load('.test_optim')
    adam.load_state_dict(load_optim)

from torch.optim import Adam
def get_optimizer(model, conf, load_optimizer):
    print(f'set optimizer {conf["type"]}')
    if conf['type'] == 'Adam':
        optim = Adam(model.parameters(), **conf['conf'])
    
    if load_optimizer:
        optim.load_state_dict(load_optimizer)
    
    return optim

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

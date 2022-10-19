import torch
from dataset import Dataset
from trainer import Trainer
from utils.optimizer import get_optimizer, get_scheduler
from utils.parse_config import get_train_config
from utils.model_related import get_model
from utils.logger import Logger
import warnings
from torch.profiler import profile, record_function, ProfilerActivity
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    args, conf = get_train_config()
    load_model = {'iters': 0,
                  'args': None,
                  'conf': None,
                  'model': None,
                  'optimizer': None}
    if args.resume:
        print(f'load model from {args.resume}')
        load_model = torch.load(args.resume)

    model = get_model(conf=conf,
                      resume=load_model['model'],
                      device=args.device)
    train_dataloader = Dataset(feature_dir=args.features, data=args.train, conf=conf['dataset'], extract_feature_online=args.extract_feature_online, device=args.device, stage='train').get_dataloader()
    dev_dataloader = Dataset(feature_dir=args.features, data=args.dev, train_data=args.train, conf=conf['dataset'], extract_feature_online=args.extract_feature_online, device=args.device, stage='dev').get_dataloader()
    optimizer = get_optimizer(model, conf=conf.get('optimizer', {}), load_optimizer=load_model['optimizer'])
    scheduler = get_scheduler(optimizer, conf.get('scheduler', {}))
    iter_logger = Logger(exp=args.exp, args=args, conf=conf, log_name='train')
    dev_logger = Logger(exp=args.exp, args=args, conf=conf, log_name='dev')
    trainer = Trainer(args=args,
                    conf=conf['trainer'],
                    iter_dataloader=train_dataloader,
                    dev_dataloader=dev_dataloader,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    iter_logger=iter_logger,
                    dev_logger=dev_logger,
                    iter=load_model['iters'])
    trainer.train()

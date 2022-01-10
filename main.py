import time
import torch
from dataset import Dataset
from trainer import Trainer
from utils.optimizer import get_optimizer
from utils.parse_config import get_train_config
from utils.model_related import get_model
from utils.logger import Logger
from yaml.events import DocumentEndEvent


if __name__ == '__main__':
    args, conf = get_train_config()
    load_model = {'iters': 0,
                  'args': None,
                  'conf': None,
                  'model': None,
                  'optimizer': None}
    if args.resume:
        print(f'read model from {args.resume}')
        load_model = torch.load(args.resume)

    model = get_model(model=conf['model'],
                      model_conf=conf['net'],
                      resume=load_model['model'],
                      device=args.device)
    train_dataloader = Dataset(feature_dir=args.features, exp=args.exp, data=args.train, conf=conf['dataset'], device=args.device).get_dataloader()
    dev_dataloader = Dataset(feature_dir=args.features, exp=args.exp, data=args.dev, conf=conf['dataset'], device=args.device).get_dataloader()
    optimizer = get_optimizer(model, conf=conf['optimizer'], load_optimizer=load_model['optimizer'])
    iter_logger = Logger(exp=args.exp, args=args, conf=conf, log_name='train')
    dev_logger = Logger(exp=args.exp, args=args, conf=conf, log_name='dev')

    trainer = Trainer(args=args,
                      conf=conf['trainer'],
                      iter_dataloader=train_dataloader,
                      dev_dataloader=dev_dataloader,
                      model=model,
                      optimizer=optimizer,
                      iter_logger=iter_logger,
                      dev_logger=dev_logger,
                      iter=load_model['iters'])
    trainer.train()

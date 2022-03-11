import os
import glob
import torch
from dataset import Dataset
from trainer import Trainer
from utils.optimizer import get_optimizer
from utils.parse_config import get_test_config
from utils.model_related import get_model
from utils.logger import Logger


if __name__ == '__main__':
    args, conf = get_test_config()
    load_model = {'iters': 0,
                  'args': None,
                  'conf': None,
                  'model': None,
                  'optimizer': None}
    model = get_model(conf=conf,
                      resume=load_model['model'],
                      device=args.device)
    test_dataloader = Dataset(feature_dir=args.features, data=args.test, conf=conf['dataset'], extract_feature_online=args.extract_feature_online, device=args.device).get_dataloader()
    optimizer = get_optimizer(model, conf=conf['optimizer'], load_optimizer=load_model['optimizer'])
    test_logger = Logger(exp=args.exp, args=args, conf=conf['logger'], log_name='test')

    trainer = Trainer(args=args,
                      conf=conf['trainer'],
                      iter_dataloader=test_dataloader,
                      dev_dataloader=None,
                      model=model,
                      optimizer=optimizer,
                      iter_logger=test_logger,
                      dev_logger=None,
                      iter=load_model['iters'])

    for checkpoint in sorted(glob.glob(os.path.join(args.exp, 'result', 'snapshot.*')), key=lambda x: int(x.split('.')[-1])):
        it = int(checkpoint.split('.')[-1])
        trainer.model = model
        if it >= int(args.start) and it <= int(args.end):
            print(f'load {checkpoint}')
            load_model = torch.load(checkpoint)
            model = get_model(conf=conf,
                              resume=load_model['model'],
                              device=args.device)
            model.eval()
            trainer.model = model
            print(f'iteration: {it}')
            trainer.test(it)
    #test_logger.write_best()
import math
import torch
from utils.model_related import save_model
class Trainer():
    def __init__(self, args, conf, iter_dataloader, dev_dataloader, model, optimizer, iter_logger, dev_logger, iter):
        print('set trainer')
        self.args = args
        self.conf = conf
        self.iter_dataloader = iter_dataloader
        self.dev_dataloader = dev_dataloader
        self.model = model
        self.iters = iter
        self.iter_logger = iter_logger
        self.dev_logger = dev_logger
        self.optimizer = optimizer
        self.accum = 0
        self.accum_grad = conf['accum_grad']
        self.iterations = conf['iterations']
        self.iteration_type = conf['iteration_type']
        self.save_per_iterations = conf['save_per_iterations']
        self.eval_per_iterations = conf['eval_per_iterations']
        self.exp = args.exp
        
    def train(self):
        iter_type_iterations = (self.iteration_type == 'iterations')
        while self.iters < self.iterations:
            for x, y, id in self.iter_dataloader:
                pred, loss = self.iter_forward(x, y)
                (loss['overall_loss'] / self.accum_grad).backward()
                self.handle_accumulate_grad()
                self.iter_logger.register_one_record(loss, x[0].shape[0])

                if iter_type_iterations:
                    if self.iteration_increase_and_check_break():
                        break

            if not iter_type_iterations:
                self.iteration_increase_and_check_break()

    def dev(self):
        for x, y, id in self.dev_dataloader:
            pred, loss = self.iter_forward(x, y)
            self.dev_logger.register_one_record(loss, x[0].shape[0])
        self.dev_logger.log_and_clear_record(self.iters)

    def test(self):
        with torch.no_grad():
            for x, y, id in self.iter_dataloader:
                pred, loss = self.iter_forward(x, y)
                self.iter_logger.write(id, y, pred)

    def handle_accumulate_grad(self):
        self.accum += 1
        if self.accum_grad == self.accum:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            if math.isnan(grad_norm):
                print("grad norm is nan. Do not update model.")
            elif math.isinf(grad_norm):
                print("grad norm in inf. Do not update model.")
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.accum = 0

    def iteration_increase_and_check_break(self):
        self.iters += 1
        if self.iters % self.save_per_iterations == 0:
            save_model(self)
        if self.iters % self.eval_per_iterations == 0:
            self.iter_logger.log_and_clear_record(self.iters)
            if self.dev_dataloader:
                with torch.no_grad():
                    self.model.eval()
                    self.dev()
                    self.model.train()
        
        if self.iters >= self.iterations:
            return True
        return False


    def iter_forward(self, x, y):
        return self.model(x, y)

    def dev_forward(self, x, y):
        with torch.no_grad():
            return self.model(x, y)

    def infer_forward(self, x):
        return self.model.inference(x)



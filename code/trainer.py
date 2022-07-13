import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tensorboardX import SummaryWriter

class Trainer():
    def __init__(self, config, model,criterion):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.writer = SummaryWriter(log_dir=config.logdir)
        self.criterion = criterion
        self.model = model.to(device)
        self.logger.info(model)
        self.load_state_dict()
        self.epoch = 0
        self.record_set = {'bpp':[],'mse':[],'sum_loss':[]}

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger

    def load_state_dict(self):
        """selectively load model
        """
        if self.config.init_ckpt=='':
            self.logger.info('Random initialization.')
        else:
            ckpt = torch.load(self.config.init_ckpt)
            self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)
        return

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, 
            os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))
        return

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items(): 
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items(): 
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))   
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  
        return 

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        for _, images in enumerate(tqdm(dataloader)):
            images = images.to(device)
            out_set = self.model(images)
            out_criterion = self.criterion(out_set, images)
            # record
            self.record_set['bpp'].append(out_criterion['bpp_loss'].item())
            self.record_set['mse'].append(out_criterion['mse_loss'].item())
            self.record_set['sum_loss'].append(out_criterion['loss'].item())
            torch.cuda.empty_cache()# empty cache.
        self.record(main_tag=main_tag, global_step=self.epoch)
        return 

    def train(self, dataloader):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        # optimizer
        self.optimizer = self.set_optimizer()
        self.logger.info('lmbda:' + str(round(self.config.lmbda,2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))
        start_time = time.time()
        for batch_step, images in enumerate(tqdm(dataloader)):
            images = images.to(device)
            self.optimizer.zero_grad()
            out_set = self.model(images)
            out_criterion = self.criterion(out_set, images)
            out_criterion["loss"].backward()
            self.optimizer.step()
            #record
            with torch.no_grad():
                self.record_set['bpp'].append(out_criterion['bpp_loss'].item())
                self.record_set['mse'].append(out_criterion['mse_loss'].item())
                self.record_set['sum_loss'].append(out_criterion['loss'].item())
                if (time.time() - start_time) > self.config.check_time*60:
                    self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
                    self.save_model()
                    start_time = time.time()
            torch.cuda.empty_cache()# empty cache.
        with torch.no_grad(): self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
        self.save_model()
        self.epoch += 1
        return

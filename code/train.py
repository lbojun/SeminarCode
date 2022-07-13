from ast import parse
import  os, argparse
from data_loader import VimeoDataset,KodakDataset
from model import factorized
from trainer import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader
from loss import RateDistortionLoss
def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default='/data1/liubj/vimeo_1')
    parser.add_argument("--lmbda", type=float, default=1e-2, help="weights for distoration.")
    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_dataset", default='/data1/liubj/Kodak24')
    parser.add_argument("--test_batch_size", type=int, default=24)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--check_time", type=float, default=10,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='tp', help="prefix of checkpoints/logger, etc.")

    args = parser.parse_args()
    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, lmbda, lr, check_time):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.lmbda = lmbda
        self.lr = lr
        self.check_time=check_time


if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('./logs', args.prefix), 
                            ckptdir=os.path.join('./ckpts', args.prefix), 
                            init_ckpt=args.init_ckpt,
                            lmbda=args.lmbda,
                            lr=args.lr, 
                            check_time=args.check_time)
    # model
    model = factorized()

    criterion = RateDistortionLoss(lmbda=args.lmbda)
    # trainer    
    trainer = Trainer(config=training_config, model=model,criterion=criterion)
    
    # dataset
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(256), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor()]
    )

    train_dataset = VimeoDataset(args.dataset, 'train', train_transforms) 
    test_dataset = KodakDataset(args.test_dataset, test_transforms) 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # training
    for epoch in range(0, args.epoch):
        if epoch>0: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr 
        trainer.train(train_dataloader)
        trainer.test(test_dataloader)
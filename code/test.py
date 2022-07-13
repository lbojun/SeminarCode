import torch
import numpy as np
import os
import pandas as pd
from model import factorized
from data_loader import KodakDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from coder import Coder
import time
from tqdm import tqdm
from data_utils import write_image, crop, cal_psnr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def test(test_dataloader, ckptdir_list, outdir, resultdir):
    # load data
    start_time = time.time()
    # load model
    model = factorized().to(device)
    for i, images in enumerate(tqdm(test_dataloader)):
        #对测试图像分辨率做一个判断
        images = crop(images)
        x = images.to(device)
        H, W = x.shape[-2:]
        print(H, W)
        num_pixel = H*W
        if not os.path.exists(outdir): os.makedirs(outdir)
        filename = os.path.join(outdir, str(i))
        print('output filename:\t', filename)
        #print('Loading Time:\t', round(time.time() - start_time, 4), 's')
        for idx, ckptdir in enumerate(ckptdir_list):
            print('='*10, f'{idx+1}', '='*10)
            # load checkpoints
            assert os.path.exists(ckptdir)
            ckpt = torch.load(ckptdir)
            model.load_state_dict(ckpt['model'])
            print('load checkpoint from \t', ckptdir)
            coder = Coder(model=model, filename=filename)

            # postfix: rate index
            postfix_idx = '_r'+str(idx+1)

             # encode
            start_time = time.time()
            _ = coder.encode(x, postfix=postfix_idx)
            print('Enc Time:\t', round(time.time() - start_time, 3), 's')
            time_enc = round(time.time() - start_time, 3)

            # decode
            start_time = time.time()
            x_dec = coder.decode(postfix=postfix_idx)  #从文件中读取数据然后解码，解码后对x_dec限定范围[0, 1]
            x_dec = torch.clamp(x_dec, min=0.0, max=1.0)
            print(x_dec.shape)
            print('Dec Time:\t', round(time.time() - start_time, 3), 's')
            time_dec = round(time.time() - start_time, 3)

            # bitrate
            bits = np.array([os.path.getsize(filename + postfix_idx + postfix)*8 \
                                    for postfix in ['_F.bin', '_H.bin']])
            bpps = (bits/num_pixel).round(3)
            print('num_pixel:',num_pixel)
            print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

            # distortion
            #重建图片
            start_time = time.time()
            # print(x_dec.detach().cpu().numpy().squeeze().shape)
            write_image(filename+postfix_idx+'_dec.png', x_dec.detach().cpu().numpy().squeeze())
            print('Write image Time:\t', round(time.time() - start_time, 3), 's')
            #计算失真PSNR
            psnr = cal_psnr(x, x_dec=x_dec) 
            print(f'psnr:  {psnr}')
            # save results
            results ={}
            results["num_points(input)"] = num_pixel
            results["num_points(output)"] = len(x_dec)
            results["bits"] = sum(bits).round(3)
            results["bits"] = sum(bits).round(3)
            results["bpp"] = sum(bpps).round(3)
            results["bpp(coords)"] = bpps[0]
            results["bpp(feats)"] = bpps[1]
            results['psnr'] = psnr
            results["time(enc)"] = time_enc
            results["time(dec)"] = time_dec
        # #results保存为csv表格文件
        #     if idx==0:
        #         df = pd.DataFrame([results])
        #     else:
        #         df1 = pd.DataFrame([results])
        #         df.append(df1)
        # if not os.path.exists(resultdir): os.makedirs(resultdir)
        # csv_name = os.path.join(resultdir, f'_pic{i}' + '.csv')
        # df.to_csv(csv_name, index=False)
        # print('Wrile results to: \t', csv_name)

    return results
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outdir", default='./output')
    parser.add_argument("--resultdir", default='./results')
    parser.add_argument("--dataset_path", default='/data1/liubj/Kodak24')
    parser.add_argument("--test_batch_size", default=1)

    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    # if not os.path.exists(args.filedir): os.makedirs(args.filedir)
    #ckptdir_list = ['./ckpts/r1_0.025bpp.pth', './ckpts/r2_0.05bpp.pth', 
                   # './ckpts/r3_0.10bpp.pth', './ckpts/r4_0.15bpp.pth', 
                    #'./ckpts/r5_0.25bpp.pth', './ckpts/r6_0.3bpp.pth', 
                    #'./ckpts/r7_0.4bpp.pth']
    ckptdir_list = ['./ckpts/tp/epoch_35.pth']

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    test_dataset = KodakDataset(args.dataset_path, test_transforms) 
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    all_results = test(test_dataloader, ckptdir_list, args.outdir, args.resultdir)
'''
    # plot RD-curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
            label="D1", marker='x', color='red')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
            label="D2", marker='x', color='blue')
    filename = os.path.split(args.filedir)[-1][:-4]
    plt.title(filename)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(args.resultdir, filename+'.jpg'))
'''


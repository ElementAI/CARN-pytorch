import os
import math
from decimal import Decimal

from . import utility
from src import backbones

import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.utils as utils
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

class Trainer():
    def __init__(self, exp_dict):
        self.exp_dict = exp_dict
        self.scale = self.exp_dict["dataset"]["scale"]
        self.rgb_range = self.exp_dict["dataset"]["rgb_range"]
        # self.model = my_model
        # self.loss = my_loss
        self.backbone = backbones.get_backbone(exp_dict)
        self.optimizer = Adam(self.backbone.parameters(),
                                lr=self.exp_dict["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
            step_size=self.exp_dict["lr_decay"],
            gamma=self.exp_dict["gamma"])
        self.ngpu = self.exp_dict["ngpu"]
        self.error_last = 1e8

    def train_on_loader(self, dataloader):
        self.backbone.train()
        losses = []

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx_scale) in enumerate(tqdm(dataloader)):
            lr, hr = self.prepare(lr, hr)

            self.optimizer.zero_grad()
            sr = self.backbone(lr, 0)
            loss = F.l1_loss(sr, hr)

            loss.backward()
            losses.append(float(loss))
            # if self.args.gclip > 0:
            #     utils.clip_grad_value_(
            #         self.model.parameters(),
            #         self.args.gclip
            #     )
            self.optimizer.step()

            if (batch + 1) % 20 == 0:
                print("loss", float(loss))

        return dict(train_loss=float(np.mean(losses)))

    def test_on_loader(self, dataloader):
        self.backbone.eval()
        results = {}
        with torch.no_grad():
            for idx_scale, scale in enumerate(tqdm(self.scale)):
                eval_psnr = 0
                # eval_ssim = 0
                dataloader.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(dataloader, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.backbone(lr, idx_scale)

                    sr = utility.quantize(sr, self.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, scale, self.rgb_range,
                            benchmark=dataloader.dataset.benchmark
                        )
                        # eval_ssim += utility.calc_ssim(sr, hr)
                        # save_list.extend([lr, hr])
                        save_list.extend([hr])

                    # if self.exp_dict["save_results"]:
                    #     self.ckp.save_results(filename, save_list, scale)
                results["val_psnr_x%d" %(scale)] = float(eval_psnr / len(dataloader))
            return results
                # self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                # mean_ssim = eval_ssim / len(self.loader_test)
                
                # best = self.ckp.log.max(0)
                # self.ckp.write_log(
                #     '[{} x{}]\tPSNR: {:.3f} (Best PSNR: {:.3f} @epoch {})'.format(
                #         self.args.data_test,
                #         scale,
                #         self.ckp.log[-1, idx_scale],
                #         best[0][idx_scale],
                #         best[1][idx_scale] + 1
                #     )
                # )

        # if not self.test_only:
        #     self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if False else 'cuda')
        def _prepare(tensor):
            if self.exp_dict["backbone"]["precision"] == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def get_state_dict(self):
        state_dict = {}
        state_dict["backbone"] = self.backbone.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["scheduler"] = self.scheduler.state_dict()
        return state_dict

    def set_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict["backbone"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

    # def terminate(self):
    #     if self.test_only:
    #         self.test()
    #         return True
    #     else:
    #         epoch = self.scheduler.last_epoch + 1
    #         return epoch >= self.args.epochs


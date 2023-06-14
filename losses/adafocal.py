'''
This implements the Adafocal loss function proposed in the NeurIPS-2022 paper "AdaFocal: Calibration-aware Adaptive Focal Loss".
Authors: Arindam Ghosh, Thomas Schaaf, and Matt Gormley.
Url: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a692a24dbc744fca340b9ba33bc6522-Abstract-Conference.html
'''

import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

class AdaFocal(nn.Module):
    def __init__(self, args, device=None):            
        super(AdaFocal, self).__init__()
        self.args = args
        self.num_bins = args.num_bins
        self.lamda = args.adafocal_lambda
        self.gamma_initial = args.adafocal_gamma_initial
        self.switch_pt = args.adafocal_switch_pt
        self.gamma_max = args.adafocal_gamma_max
        self.gamma_min = args.adafocal_gamma_min
        self.update_gamma_every = args.update_gamma_every
        self.device = device
        # This initializes the bin_stats variable
        self.bin_stats = collections.defaultdict(dict)
        for bin_no in range(self.num_bins):
            self.bin_stats[bin_no]['lower_boundary'] = bin_no*(1/self.num_bins)
            self.bin_stats[bin_no]['upper_boundary'] = (bin_no+1)*(1/self.num_bins)
            self.bin_stats[bin_no]['gamma'] = self.gamma_initial

    # This function updates the bin statistics which are used by the Adafocal loss at every epoch.
    def update_bin_stats(self, val_adabin_dict):
        for bin_no in range(self.num_bins):
            # This is the Adafocal gamma update rule
            prev_gamma = self.bin_stats[bin_no]['gamma']
            exp_term = val_adabin_dict[bin_no]['calibration_gap']
            if prev_gamma > 0:
                next_gamma = prev_gamma * math.exp(self.lamda*exp_term)
            else:
                next_gamma = prev_gamma * math.exp(-self.lamda*exp_term)    
            # This switches between focal and inverse-focal loss when required.
            if abs(next_gamma) < self.switch_pt:
                if next_gamma > 0:
                    next_gamma = -self.switch_pt
                else:
                    next_gamma = self.switch_pt
            self.bin_stats[bin_no]['gamma'] = max(min(next_gamma, self.gamma_max), self.gamma_min) # gamma-clipping
            self.bin_stats[bin_no]['lower_boundary'] = val_adabin_dict[bin_no]['lower_bound']
            self.bin_stats[bin_no]['upper_boundary'] = val_adabin_dict[bin_no]['upper_bound']
        # This saves the "bin_stats" to a text file.
        save_file = os.path.join(self.args.save_path, "val_bin_stats.txt")
        with open(save_file, "a") as write_file:
            json.dump(self.bin_stats, write_file)
            write_file.write("\n")
        return

    # This function selects the gammas for each sample based on which bin it falls into.
    def get_gamma_per_sample(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            for bin_no, stats in self.bin_stats.items():
                if bin_no==0 and pt_sample < stats['upper_boundary']:
                    break
                elif bin_no==self.num_bins-1 and pt_sample >= stats['lower_boundary']:
                    break
                elif pt_sample >= stats['lower_boundary'] and pt_sample < stats['upper_boundary']:
                    break
            gamma_list.append(stats['gamma'])
        return torch.tensor(gamma_list).to(self.device)

    # This computes the loss value to be returned for back-propagation.
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        gamma = self.get_gamma_per_sample(pt)
        gamma_sign = torch.sign(gamma)
        gamma_mag = torch.abs(gamma)
        pt = gamma_sign * pt
        loss = -1 * ((1 - pt + 1e-20)**gamma_mag) * logpt # 1e-20 added for numerical stability 
 
        return loss.sum()
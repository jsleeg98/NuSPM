# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import time
from copy import deepcopy
from functools import wraps
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from . import logger as log
from . import utils
from .logger import TrainingMetrics, ValidationMetrics
from .models.common import EMA
import matplotlib.pyplot as plt


resnet50_list = [118013952,
                 12845056, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224,
                 102760448, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224,
                 51380224, 115605504, 51380224,
                 102760448, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224,
                 51380224, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224,
                 102760448, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224]

resnet50_mac_list = [614656.0,
                     3136.0, 28224.0, 3136.0, 3136.0, 28224.0, 3136.0, 3136.0, 28224.0, 3136.0,
                     3136.0, 7056.0, 784.0, 784.0, 7056.0, 784.0, 784.0, 7056.0, 784.0, 784.0, 7056.0, 784.0,
                     784.0, 1764.0, 196.0, 196.0, 1764.0, 196.0, 196.0, 1764.0, 196.0, 196.0, 1764.0, 196.0,
                     196.0, 1764.0, 196.0, 196.0, 1764.0, 196.0,
                     196.0, 441.0, 49.0, 49.0, 441.0, 49.0, 49.0, 441.0, 49.0]

resnet50_list_no_share = [118013952,
                 12845056, 115605504, 51380224, 115605504, 51380224, 115605504,
                 102760448, 115605504, 51380224, 115605504, 51380224, 115605504, 51380224, 115605504,
                 102760448, 115605504, 51380224, 115605504, 51380224, 115605504,
                 51380224, 115605504, 51380224, 115605504, 51380224, 115605504,
                 102760448, 115605504, 51380224, 115605504, 51380224, 115605504]

resnet50_mac_list_no_share = [614656.0,
                     3136.0, 28224.0, 3136.0, 28224.0, 3136.0, 28224.0,
                     3136.0, 7056.0, 784.0, 7056.0, 784.0, 7056.0, 784.0, 7056.0,
                     784.0, 1764.0, 196.0, 1764.0, 196.0, 1764.0, 196.0, 1764.0,
                     196.0, 1764.0, 196.0, 1764.0,
                     196.0, 441.0, 49.0, 441.0, 49.0, 441.0]

resnet50_channel_list = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512,
                         128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256,
                         256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]

resnet50_in_list = [3, 64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512,
                    128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256,
                    256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512]

resnet34_list = [118013952,
                 115605504, 115605504, 115605504, 115605504, 115605504, 115605504,
                 57802752, 115605504, 115605504, 115605504, 115605504, 115605504, 115605504, 115605504,
                 57802752, 115605504, 115605504, 115605504, 115605504, 115605504, 115605504, 115605504,
                 115605504, 115605504, 115605504, 115605504,
                 57802752, 115605504, 115605504, 115605504, 115605504, 115605504]

resnet34_mac_list = [614656.0,
                     28224.0, 28224.0, 28224.0, 28224.0, 28224.0, 28224.0,
                     7056.0, 7056.0, 7056.0, 7056.0, 7056.0, 7056.0, 7056.0, 7056.0,
                     1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0, 1764.0,
                     441.0, 441.0, 441.0, 441.0, 441.0, 441.0]

def append_loss_mac(model, percent=0.66, alpha=5, arch=None, shared_mac_loss_off=False):
    alpha_adjust = alpha
    Branches = torch.tensor([]).cuda()  # remain channels number
    li_share_branch = []
    for name, module in model.named_modules():
        if 'share' in name:
            w = module.weight.detach()
            binary_w = (w > 0.5).float()
            residual = w - binary_w
            branch_out = module.weight - residual
            li_share_branch.append(torch.sum(torch.squeeze(branch_out)))
        elif 'scale' in name:
            w = module.weight.detach()
            binary_w = (w > 0.5).float()
            residual = w - binary_w
            branch_out = module.weight - residual
            Branches = torch.cat((Branches, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)
    if not shared_mac_loss_off:
        if arch == 'resnet50_SPM':
            # insert SPM in layer1
            for idx in [3, 6, 9]:
                Branches = torch.cat((Branches[:idx], li_share_branch[0].reshape(1), Branches[idx:]))

            # insert SPM in layer2
            for idx in [12, 15, 18, 21]:
                Branches = torch.cat((Branches[:idx], li_share_branch[1].reshape(1), Branches[idx:]))

            # insert SPM in layer3
            for idx in [24, 27, 30, 33, 36, 39]:
                Branches = torch.cat((Branches[:idx], li_share_branch[2].reshape(1), Branches[idx:]))

            # insert SPM in layer4
            for idx in [42, 45, 48]:
                Branches = torch.cat((Branches[:idx], li_share_branch[3].reshape(1), Branches[idx:]))

            target_macs = torch.tensor(sum(resnet50_list) / 1e9).cuda()
            in_channel = torch.cat((torch.tensor([3]).cuda(), Branches[:-1]), dim=0)

            current_macs = torch.sum(torch.tensor(in_channel) * torch.tensor(resnet50_mac_list).cuda() * Branches) / 1e9
            criterion = nn.MSELoss()
            branch_loss = criterion(current_macs, target_macs * (1 - percent))
            current_macs_ratio = float(current_macs / target_macs * 100)
        elif arch == 'resnet34_SPM':
            for idx in [2, 4, 6]:
                Branches = torch.cat((Branches[:idx], li_share_branch[0].reshape(1), Branches[idx:]))

            # insert SPM in layer2
            for idx in [8, 10, 12, 14]:
                Branches = torch.cat((Branches[:idx], li_share_branch[1].reshape(1), Branches[idx:]))

            # insert SPM in layer3
            for idx in [16, 18, 20, 22, 24, 26]:
                Branches = torch.cat((Branches[:idx], li_share_branch[2].reshape(1), Branches[idx:]))

            # insert SPM in layer4
            for idx in [28, 30, 32]:
                Branches = torch.cat((Branches[:idx], li_share_branch[3].reshape(1), Branches[idx:]))

            target_macs = torch.tensor(sum(resnet34_list) / 1e9).cuda()
            in_channel = torch.cat((torch.tensor([3]).cuda(), Branches[:-1]), dim=0)

            current_macs = torch.sum(torch.tensor(in_channel) * torch.tensor(resnet34_mac_list).cuda() * Branches) / 1e9
            criterion = nn.MSELoss()
            branch_loss = criterion(current_macs, target_macs * (1 - percent))
            current_macs_ratio = float(current_macs / target_macs * 100)
    else:
        target_macs = torch.tensor(sum(resnet50_list_no_share) / 1e9).cuda()
        in_channel = torch.cat((torch.tensor([3]).cuda(), Branches[:-1]), dim=0)
        current_macs = torch.sum(
            in_channel.clone().detach() * torch.tensor(resnet50_mac_list_no_share).cuda() * Branches) / 1e9

        criterion = nn.MSELoss()
        branch_loss = criterion(current_macs, target_macs * (1 - percent))
        current_macs_ratio = float(current_macs / target_macs * 100)

    return branch_loss * alpha_adjust, current_macs_ratio


def append_loss_nuc(model, alpha=5, arch=None):
    alpha_adjust = alpha
    Branches = torch.tensor([]).cuda()
    li_conv_w = []
    li_SPM_w = []
    li_share_w = []
    for name, module in model.named_modules():
        if 'share' in name:
            w = module.weight.detach()
            binary_w = (w > 0.5).float()
            residual = w - binary_w
            branch_out = module.weight - residual
            li_share_w.append(branch_out)
        elif 'scale' in name:
            w = module.weight.detach()
            binary_w = (w > 0.5).float()
            residual = w - binary_w
            branch_out = module.weight - residual
            li_SPM_w.append(branch_out)
            Branches = torch.cat((Branches, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)
        elif isinstance(module, nn.Conv2d):
            w = module.weight.detach().clone()
            li_conv_w.append(w)

    if arch == 'resnet50_SPM':
        # insert SPM in layer1
        for idx in [3, 4, 7, 10]:
            li_SPM_w.insert(idx, li_share_w[0])

        # insert SPM in layer2
        for idx in [13, 14, 17, 20, 23]:
            li_SPM_w.insert(idx, li_share_w[1])

        # insert SPM in layer3
        for idx in [26, 27, 30, 33, 36, 39, 42]:
            li_SPM_w.insert(idx, li_share_w[2])

        # insert SPM in layer4
        for idx in [45, 46, 49, 52]:
            li_SPM_w.insert(idx, li_share_w[3])

        li_pruned_conv = []
        for conv, SPM in zip(li_conv_w, li_SPM_w):
            li_pruned_conv.append(conv * SPM)

        origin_nuc_norm = torch.tensor([]).cuda()
        pruned_nuc_norm = torch.tensor([]).cuda()
        for origin, pruned in zip(li_conv_w, li_pruned_conv):
            origin_nuc_norm = torch.cat((origin_nuc_norm, torch.unsqueeze(torch.norm(torch.flatten(origin, start_dim=1), p='nuc'), dim=0)), dim=0)
            pruned_nuc_norm = torch.cat((pruned_nuc_norm, torch.unsqueeze(torch.norm(torch.flatten(pruned, start_dim=1), p='nuc'), dim=0)), dim=0)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        nuc_loss = criterion(origin_nuc_norm, pruned_nuc_norm)


    elif arch == 'resnet34_SPM':
        # insert SPM in layer1
        for idx in [2, 3, 4, 6]:
            li_SPM_w.insert(idx, li_share_w[0])

        # insert SPM in layer2
        for idx in [9, 10, 12, 14, 16]:
            li_SPM_w.insert(idx, li_share_w[1])

        # insert SPM in layer3
        for idx in [18, 19, 21, 23, 25, 27, 29]:
            li_SPM_w.insert(idx, li_share_w[2])

        # insert SPM in layer4
        for idx in [31, 32, 34, 36]:
            li_SPM_w.insert(idx, li_share_w[3])

        li_pruned_conv = []
        for conv, SPM in zip(li_conv_w, li_SPM_w):
            li_pruned_conv.append(conv * SPM)

        origin_nuc_norm = torch.tensor([]).cuda()
        pruned_nuc_norm = torch.tensor([]).cuda()
        for origin, pruned in zip(li_conv_w, li_pruned_conv):
            origin_nuc_norm = torch.cat(
                (origin_nuc_norm, torch.unsqueeze(torch.norm(torch.flatten(origin, start_dim=1), p='nuc'), dim=0)),
                dim=0)
            pruned_nuc_norm = torch.cat(
                (pruned_nuc_norm, torch.unsqueeze(torch.norm(torch.flatten(pruned, start_dim=1), p='nuc'), dim=0)),
                dim=0)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        nuc_loss = criterion(origin_nuc_norm, pruned_nuc_norm)

    return nuc_loss * alpha_adjust


class Executor:
    def __init__(
        self,
        model: nn.Module,
        loss: Optional[nn.Module],
        cuda: bool = True,
        memory_format: torch.memory_format = torch.contiguous_format,
        amp: bool = False,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        divide_loss: int = 1,
        ts_script: bool = False,
        mac_loss_off: bool = False,
        alpha_mac: float = 0.1,
        target_ratio: float = 0.3,
        nuc_loss_off: bool = False,
        alpha_nuc: float = 0.1,
        writer=None,
        arch=None,
        shared_mac_loss_off=False,
        args=None,
    ):
        assert not (amp and scaler is None), "Gradient Scaler is needed for AMP"

        def xform(m: nn.Module) -> nn.Module:
            if cuda:
                m = m.cuda()
            m.to(memory_format=memory_format)
            return m

        self.model = xform(model)
        if ts_script:
            self.model = torch.jit.script(self.model)
        self.ts_script = ts_script
        self.loss = xform(loss) if loss is not None else None
        self.amp = amp
        self.scaler = scaler
        self.is_distributed = False
        self.divide_loss = divide_loss
        self._fwd_bwd = None
        self._forward = None
        self.mac_loss_off = mac_loss_off
        self.alpha_mac = alpha_mac
        self.target_ratio = target_ratio
        self.cur_macs_ratio = 100.
        self.mac_loss = 0.  # record mac_loss
        self.nuc_loss_off = nuc_loss_off
        self.alpha_nuc = alpha_nuc
        self.nuc_loss = 0.  # record nuc_loss
        self.cls_loss = 0.  # record cls_loss
        self.total_loss = 0.  # record total_loss
        self.writer = writer
        self.arch = arch
        self.shared_mac_loss_off = shared_mac_loss_off
        self.args = args

    def distributed(self, gpu_id):
        self.is_distributed = True
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)  # for S_max
        torch.cuda.current_stream().wait_stream(s)

    def _fwd_bwd_fn(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with autocast(enabled=self.amp):
            cls_loss = self.loss(self.model(input), target)
            cls_loss /= self.divide_loss
            self.cls_loss += cls_loss

        if not self.mac_loss_off:
            mac_loss, current_macs_ratio = append_loss_mac(self.model, percent=self.target_ratio, alpha=self.alpha_mac, arch=self.arch, shared_mac_loss_off=self.shared_mac_loss_off)
            mac_loss /= self.divide_loss
            self.cur_macs_ratio = current_macs_ratio
            self.mac_loss += mac_loss
        else:
            # _, current_macs_ratio = append_loss_mac(self.model, percent=self.target_ratio, alpha=self.alpha_mac,
            #                                                arch=self.arch)
            # self.cur_macs_ratio = current_macs_ratio
            mac_loss = torch.tensor(0.)

        if not self.nuc_loss_off:
            nuc_loss = append_loss_nuc(self.model, alpha=self.alpha_nuc, arch=self.arch)
            nuc_loss /= self.divide_loss
            self.nuc_loss += nuc_loss
        else:
            nuc_loss = torch.tensor(0.)

        reg_loss = nuc_loss + mac_loss
        loss = cls_loss + reg_loss
        self.total_loss += loss

        self.scaler.scale(loss).backward()

        return loss

    def _forward_fn(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), autocast(enabled=self.amp):
            output = self.model(input)
            loss = None if self.loss is None else self.loss(output, target)

        return output if loss is None else loss, output

    def optimize(self, fn):
        return fn

    @property
    def forward_backward(self):
        if self._fwd_bwd is None:
            if self.loss is None:
                raise NotImplementedError(
                    "Loss must not be None for forward+backward step"
                )
            self._fwd_bwd = self.optimize(self._fwd_bwd_fn)
        return self._fwd_bwd

    @property
    def forward(self):
        if self._forward is None:
            self._forward = self.optimize(self._forward_fn)
        return self._forward

    def train(self):
        self.model.train()
        if self.loss is not None:
            self.loss.train()

    def eval(self):
        self.model.eval()
        if self.loss is not None:
            self.loss.eval()


class Trainer:
    def __init__(
        self,
        executor: Executor,
        optimizer: torch.optim.Optimizer,
        grad_acc_steps: int,
        ema: Optional[float] = None,
    ):
        self.executor = executor
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps
        self.use_ema = False
        if ema is not None:
            self.ema_executor = deepcopy(self.executor)
            self.ema = EMA(ema, self.ema_executor.model)
            self.use_ema = True

        self.optimizer.zero_grad(set_to_none=True)
        self.steps_since_update = 0

    def train(self):
        self.executor.train()
        if self.use_ema:
            self.ema_executor.train()

    def eval(self):
        self.executor.eval()
        if self.use_ema:
            self.ema_executor.eval()

    def train_step(self, input, target, step=None, writer=None):
        global iteration_index
        loss = self.executor.forward_backward(input, target)
        self.steps_since_update += 1

        if self.steps_since_update == self.grad_acc_steps:
            if self.executor.scaler is not None:
                self.executor.scaler.step(self.optimizer)
                self.executor.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps_since_update = 0

        torch.cuda.synchronize()

        if self.use_ema:
            self.ema(self.executor.model, step=step)

        if iteration_index <= 1000:
            writer.add_scalar("macs/cur macs ratio (1000)", self.executor.cur_macs_ratio, iteration_index)
        if iteration_index <= 5000:
            writer.add_scalar("macs/cur macs ratio (5000)", self.executor.cur_macs_ratio, iteration_index)


        writer.add_scalar("macs/cur macs ratio", self.executor.cur_macs_ratio, iteration_index)
        iteration_index += 1

        return loss

    def validation_steps(self) -> Dict[str, Callable]:
        vsd: Dict[str, Callable] = {"val": self.executor.forward}
        if self.use_ema:
            vsd["val_ema"] = self.ema_executor.forward
        return vsd

    def state_dict(self) -> dict:
        res = {
            "state_dict": self.executor.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_ema:
            res["state_dict_ema"] = self.ema_executor.model.state_dict()

        return res


def train(
    train_step,
    train_loader,
    lr_scheduler,
    grad_scale_fn,
    log_fn,
    timeout_handler,
    prof=-1,
    step=0,
    writer=None,
    args=None,
):
    interrupted = False

    end = time.time()

    data_iter = enumerate(train_loader)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr = lr_scheduler(i)
        data_time = time.time() - end

        loss = train_step(input, target, step=step + i, writer=writer)
        it_time = time.time() - end

        with torch.no_grad():
            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.detach())
            else:
                reduced_loss = loss.detach()

        log_fn(
            compute_ips=utils.calc_ips(bs, it_time - data_time),
            total_ips=utils.calc_ips(bs, it_time),
            data_time=data_time,
            compute_time=it_time - data_time,
            lr=lr,
            loss=reduced_loss.item(),
            grad_scale=grad_scale_fn(),
        )

        end = time.time()
        if prof > 0 and (i + 1 >= prof):
            time.sleep(5)
            break
        if ((i + 1) % 20 == 0) and timeout_handler.interrupted:
            time.sleep(5)
            interrupted = True
            break

    return interrupted


def validate(infer_fn, val_loader, log_fn, prof=-1, with_loss=True, topk=5):
    top1 = log.AverageMeter()
    # switch to evaluate mode
    latency = 0
    end = time.time()

    data_iter = enumerate(val_loader)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        latency_s = time.time()
        if with_loss:
            loss, output = infer_fn(input, target)
        else:
            output = infer_fn(input)
        latency += time.time() - latency_s

        with torch.no_grad():
            precs = utils.accuracy(output.data, target, topk=(1, topk))

            if torch.distributed.is_initialized():
                if with_loss:
                    reduced_loss = utils.reduce_tensor(loss.detach())
                precs = map(utils.reduce_tensor, precs)
            else:
                if with_loss:
                    reduced_loss = loss.detach()

        precs = map(lambda t: t.item(), precs)
        infer_result = {f"top{k}": (p, bs) for k, p in zip((1, topk), precs)}

        if with_loss:
            infer_result["loss"] = (reduced_loss.item(), bs)

        torch.cuda.synchronize()

        it_time = time.time() - end

        top1.record(infer_result["top1"][0], bs)

        log_fn(
            compute_ips=utils.calc_ips(bs, it_time - data_time),
            total_ips=utils.calc_ips(bs, it_time),
            data_time=data_time,
            compute_time=it_time - data_time,
            **infer_result,
        )

        end = time.time()
        if (prof > 0) and (i + 1 >= prof):
            time.sleep(5)
            break


    print(f'latency : {latency}')
    return top1.get_val()


# Train loop {{{
def train_loop(
    trainer: Trainer,
    lr_scheduler,
    train_loader,
    train_loader_len,
    val_loader,
    logger,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    early_stopping_patience=-1,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
    keep_last_n_checkpoints=0,
    topk=5,
    writer=None,
    S_max=90,
    args=None,
):
    checkpointer = utils.Checkpointer(
        last_filename=checkpoint_filename,
        checkpoint_dir=checkpoint_dir,
        keep_last_n=keep_last_n_checkpoints,
    )
    train_metrics = TrainingMetrics(logger)
    val_metrics = {
        k: ValidationMetrics(logger, k, topk) for k in trainer.validation_steps().keys()
    }
    training_step = trainer.train_step

    prec1 = -1

    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    global iteration_index
    iteration_index = 0
    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    with utils.TimeoutHandler() as timeout_handler:
        interrupted = False
        for epoch in range(start_epoch, end_epoch):
            if logger is not None:
                logger.start_epoch()
            if not skip_training:
                if logger is not None:
                    data_iter = logger.iteration_generator_wrapper(
                        train_loader, mode="train"
                    )
                else:
                    data_iter = train_loader

                trainer.train()
                if epoch >= S_max:
                    trainer.executor.mac_loss_off = True
                    trainer.executor.nuc_loss_off = True
                    for name, param in trainer.executor.model.named_parameters():
                        if 'scale' in name:
                            param.requires_grad = False
                    print('freeze')

                interrupted = train(
                    training_step,
                    data_iter,
                    lambda i: lr_scheduler(trainer.optimizer, i, epoch),
                    trainer.executor.scaler.get_scale,
                    train_metrics.log,
                    timeout_handler,
                    prof=prof,
                    step=epoch * train_loader_len,
                    writer=writer,
                    args=args
                )
                print(f'cur macs ratio : {trainer.executor.cur_macs_ratio}')
                print(f'mac loss(per epoch) : {trainer.executor.mac_loss}')
                print(f'nuc loss(per epoch) : {trainer.executor.nuc_loss}')
                print(f'cls loss(per epoch : {trainer.executor.cls_loss}')
                print(f'total loss(per epoch) : {trainer.executor.total_loss}')
                writer.add_scalar("loss/mac_loss(per epoch)", trainer.executor.mac_loss, epoch)
                writer.add_scalar("loss/nuc_loss(per epoch)", trainer.executor.nuc_loss, epoch)
                writer.add_scalar("loss/cls_loss(per epoch)", trainer.executor.cls_loss, epoch)
                writer.add_scalar("loss/total_loss(per epoch)", trainer.executor.total_loss, epoch)
                writer.flush()

                trainer.executor.mac_loss = 0.  # reset for next epoch
                trainer.executor.nuc_loss = 0.  # reset for next epoch
                trainer.executor.cls_loss = 0.  # reset for next epoch
                trainer.executor.total_loss = 0.  # reset for next epoch

            if not skip_validation:
                trainer.eval()
                for k, infer_fn in trainer.validation_steps().items():
                    if logger is not None:
                        data_iter = logger.iteration_generator_wrapper(
                            val_loader, mode="val"
                        )
                    else:
                        data_iter = val_loader

                    step_prec1, _ = validate(
                        infer_fn,
                        data_iter,
                        val_metrics[k].log,
                        prof=prof,
                        topk=topk,
                    )

                    if k == "val":
                        prec1 = step_prec1

                writer.add_scalar("acc/prec1", prec1, epoch)
                print(f'prec1 : {prec1}')
                writer.flush()

                if prec1 > best_prec1:
                    is_best = True
                    best_prec1 = prec1
                else:
                    is_best = False
            else:
                is_best = False
                best_prec1 = 0

            if logger is not None:
                logger.end_epoch()

            if save_checkpoints and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "best_prec1": best_prec1,
                    **trainer.state_dict(),
                }
                if (epoch + 1) % 10 == 0 or (epoch + 1) == 65:
                    checkpointer.save_checkpoint(
                        checkpoint_state,
                        is_best,
                        filename=f"checkpoint_{epoch:04}.pth.tar",
                    )
                    torch.save(trainer.state_dict()['state_dict'], checkpointer.get_full_path(f'model_epoch_{epoch + 1}.pth'))
                if is_best:
                    checkpointer.save_checkpoint(
                        checkpoint_state,
                        is_best,
                        filename=f"checkpoint_{epoch:04}_best.pth.tar",
                    )
                    torch.save(trainer.state_dict()['state_dict'], checkpointer.get_full_path(f'model_best.pth'))

            if early_stopping_patience > 0:
                if not is_best:
                    epochs_since_improvement += 1
                else:
                    epochs_since_improvement = 0
                if epochs_since_improvement >= early_stopping_patience:
                    break
            if interrupted:
                break

            writer.add_scalar("acc/best prec1", best_prec1, epoch)
            writer.flush()
            print(f'best prec1 : {best_prec1}')

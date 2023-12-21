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
import os

os.environ[
    "KMP_AFFINITY"
] = "disabled"  # We need to do this before importing anything else as a workaround for this bug: https://github.com/pytorch/pytorch/issues/28389

import argparse

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from image_classification.dataloaders import *
from image_classification.training import *
from image_classification.utils import *
from image_classification.models import (
    resnet34,
    resnet34_SPM,
    resnet50,
    resnet50_SPM,
    resnet101,
    resnet101_SPM,
    resnext101_32x4d,
    se_resnext101_32x4d,
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_widese_b0,
    efficientnet_widese_b4,
)
import PruneHandler as PH
from ptflops import get_model_complexity_info


def available_models():
    models = {
        m.name: m
        for m in [
            resnet34,
            resnet34_SPM,
            resnet50,
            resnet50_SPM,
            resnet101,
            resnet101_SPM,
            resnext101_32x4d,
            se_resnext101_32x4d,
            efficientnet_b0,
            efficientnet_b4,
            efficientnet_widese_b0,
            efficientnet_widese_b4,
        ]
    }
    return models


def add_parser_arguments(parser, skip_arch=False):
    parser.add_argument("-data", metavar="DIR", default='../../datasets/imagenet',help="path to dataset")
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="pytorch",
        choices=DATA_BACKEND_CHOICES,
        help="data backend: "
        + " | ".join(DATA_BACKEND_CHOICES)
        + " (default: dali-cpu)",
    )
    parser.add_argument(
        "--interpolation",
        metavar="INTERPOLATION",
        default="bilinear",
        help="interpolation type for resizing images: bilinear, bicubic or triangular(DALI only)",
    )
    if not skip_arch:
        model_names = available_models().keys()
        parser.add_argument(
            "--arch",
            "-a",
            metavar="ARCH",
            default="resnet50_SPM",
            choices=model_names,
            help="model architecture: "
            + " | ".join(model_names)
            + " (default: resnet50)",
        )

    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 5)",
    )
    parser.add_argument(
        "--prefetch",
        default=2,
        type=int,
        metavar="N",
        help="number of samples prefetched by each loader",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 2500)",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )
    parser.add_argument(
        "--state_dict",
        type=str,
        default=None,
        help="state_dict path",
    )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def validate(val_loader, model, args):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)

            output = output.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            prec5 = accuracy(output.data, target, topk=(5,))[0]
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.1f}'
          .format(top1=top1))

    return top1.avg

def main():
    parser = argparse.ArgumentParser()

    add_parser_arguments(parser)

    args, rest = parser.parse_known_args()
    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )

    image_size = 224  # for ImageNet

    get_train_loader = get_pytorch_train_loader
    get_val_loader = get_pytorch_val_loader

    val_loader, val_loader_len = get_val_loader(
        args.data,
        image_size,
        args.batch_size,
        1000,
        False,
        interpolation=args.interpolation,
        workers=args.workers,
        memory_format=memory_format,
        prefetch_factor=args.prefetch,
    )

    model = resnet50_SPM()

    model_state = torch.load(args.state_dict)
    model.load_state_dict(model_state)

    print('Before reconfigure'.center(100, '-'))
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Before MACs(G): ', macs / (10 ** 9)))
    print('{:<30}  {:<8}'.format('Number of parameters(M): ', params / (10 ** 6)))

    model.cuda()
    model.to(memory_format=memory_format)
    model.eval()

    print('Validate before reconfigure model'.center(100, '-'))
    validate(val_loader, model, args)  # validate before reconfiguration
    print('-' * 100)

    print('Reconfiguring'.center(100, '-'))
    # reconfigure model
    ph = PH.PruneHandler(model)
    if args.arch == 'resnet50_SPM':
        model = ph.reconstruction_model('bottle', SPM=True, arch='resnet50')
    print('Done'.center(100, '-'))

    print('After reconfigure'.center(100, '-'))
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('After MACs(G): ', macs / (10 ** 9)))
    print('{:<30}  {:<8}'.format('Number of parameters(M): ', params / (10 ** 6)))

    model.cuda()
    model.to(memory_format=memory_format)
    model.eval()

    print('Validate after reconfigure model'.center(100, '-'))
    validate(val_loader, model, args)  # validate after reconfigureation
    print('-' * 100)

if __name__ == "__main__":
    main()

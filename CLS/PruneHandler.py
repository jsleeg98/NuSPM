import torch
from image_classification.models import resnet
from image_classification.models import common
import torch.nn as nn

class PruneHandler():
    def __init__(self, model):
        self.model = model
        self.remain_index = []
        self.union_index = []
        self.model.to('cpu')

    def get_remain_index(self):
        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Conv2d):  # for model.conv1
                tmp_remain_index = torch.where(torch.norm(module.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                # if len(tmp_remain_index) == 0:  # pruned all channels in layer
                #     tmp_remain_index = list(range(int(module.weight.shape[0] * 0.1)))  # revive 10% channel
                self.remain_index.append([tmp_remain_index])
            elif isinstance(module, torch.nn.Sequential):  # for model.layer
                li_li_remain_index = []
                for name_, module_ in module.named_children():
                    if isinstance(module_, resnet.BasicBlock):
                        li_remain_index = []
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                tmp_remain_index = torch.where(torch.norm(module__.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                                # if len(tmp_remain_index) == 0:  # pruned all channels in layer
                                #     tmp_remain_index = list(range(int(module__.weight.shape[0] * 0.1)))  # revive 10% channel
                                li_remain_index.append(tmp_remain_index)
                            elif isinstance(module__, torch.nn.Sequential):
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        tmp_remain_index = torch.where(torch.norm(module___.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                                        # if len(tmp_remain_index) == 0:  # pruned all channels in layer
                                        #     tmp_remain_index = list(range(int(module___.weight.shape[0] * 0.1)))  # revive 10% channel
                                        li_remain_index.append(tmp_remain_index)
                        li_li_remain_index.append(li_remain_index)
                    elif isinstance(module_, resnet.Bottleneck):  # for model.layer.Bottleneck
                        li_remain_index = []
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                tmp_remain_index = torch.where(torch.norm(module__.weight, p=1, dim=(1, 2, 3)) != 0)[
                                    0].tolist()
                                # if len(tmp_remain_index) == 0:  # pruned all channels in layer
                                #     tmp_remain_index = list(range(int(module__.weight.shape[0] * 0.1)))  # revive 10% channel
                                li_remain_index.append(tmp_remain_index)
                            elif isinstance(module__, torch.nn.Sequential):  # for model.layer.Bottleneck.downsample
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        tmp_remain_index = \
                                        torch.where(torch.norm(module___.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                                        # if len(tmp_remain_index) == 0:  # pruned all channels in layer
                                        #     tmp_remain_index = list(range(int(module___.weight.shape[0] * 0.1)))  # revive 10% channel
                                        li_remain_index.append(tmp_remain_index)
                        li_li_remain_index.append(li_remain_index)
                self.remain_index.append(li_li_remain_index)

    def get_remain_index_SPM(self, block):
        scale1_shared_weight = self.model.scale1_share.weight.detach()
        scale1_shared_binary = (scale1_shared_weight > 0.5).float()
        scale1_shared_remain = torch.where(scale1_shared_binary == 1)[0]
        scale2_shared_weight = self.model.scale2_share.weight.detach()
        scale2_shared_binary = (scale2_shared_weight > 0.5).float()
        scale2_shared_remain = torch.where(scale2_shared_binary == 1)[0]
        scale3_shared_weight = self.model.scale3_share.weight.detach()
        scale3_shared_binary = (scale3_shared_weight > 0.5).float()
        scale3_shared_remain = torch.where(scale3_shared_binary == 1)[0]
        scale4_shared_weight = self.model.scale4_share.weight.detach()
        scale4_shared_binary = (scale4_shared_weight > 0.5).float()
        scale4_shared_remain = torch.where(scale4_shared_binary == 1)[0]

        if block == 'basic':
            for name, module in self.model.named_modules():
                if isinstance(module, common.BinaryConv2d):
                    w = module.weight.detach()
                    binary_w = (w > 0.5).float()
                    residual = w - binary_w
                    branch_out = module.weight - residual
                    tmp_remain_index = torch.where(branch_out.squeeze() == 1)[0]
                    self.remain_index.append(tmp_remain_index)
                    if 'scale1' in name:
                        if 'layers.0' in name:
                            self.remain_index.append(scale1_shared_remain)
                        elif 'layers.1' in name:
                            self.remain_index.append(scale2_shared_remain)
                        elif 'layers.2' in name:
                            self.remain_index.append(scale3_shared_remain)
                        elif 'layers.3' in name:
                            self.remain_index.append(scale4_shared_remain)
            del self.remain_index[1:5]
        elif block == 'bottle':
            for name, module in self.model.named_modules():
                if isinstance(module, common.BinaryConv2d):
                    w = module.weight.detach()
                    binary_w = (w > 0.5).float()
                    residual = w - binary_w
                    branch_out = module.weight - residual
                    tmp_remain_index = torch.where(branch_out.squeeze() == 1)[0]
                    self.remain_index.append(tmp_remain_index)
                    if 'scale2' in name:
                        if 'layers.0' in name:
                            self.remain_index.append(scale1_shared_remain)
                        elif 'layers.1' in name:
                            self.remain_index.append(scale2_shared_remain)
                        elif 'layers.2' in name:
                            self.remain_index.append(scale3_shared_remain)
                        elif 'layers.3' in name:
                            self.remain_index.append(scale4_shared_remain)
            del self.remain_index[1:5]

    def reconsruction_bottle_SPM(self, arch):
        li_in_channels_remain_index = self.remain_index[:-1]
        li_in_channels_remain_index.insert(0, torch.tensor([0, 1, 2]))
        li_out_channels_remain_index = self.remain_index[:]

        if arch == 'resnet34':
            pruned_model = resnet.resnet34()
            pruned_model.load_state_dict(self.model.state_dict(), strict=False)

            idx = 0
            for name, module in pruned_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    if 'downsample' in name:
                        idx -= 1  # follow conv2
                        module.weight = torch.nn.parameter.Parameter(torch.index_select(
                            module.weight, 1, li_in_channels_remain_index[idx - 1]))  # follow conv1 in_channels
                        module.weight = torch.nn.parameter.Parameter(torch.index_select(
                            module.weight, 0, li_out_channels_remain_index[idx]))  # follow conv2 out_channels
                        module.in_channels = len(li_in_channels_remain_index[idx - 1])
                        module.out_channels = len(li_out_channels_remain_index[idx])
                    else:
                        if len(li_in_channels_remain_index[idx]) == 0:  # pruned all in filters
                            li_in_channels_remain_index[idx] = torch.tensor([0])
                        if len(li_out_channels_remain_index[idx]) == 0:  # pruned all out filters
                            li_out_channels_remain_index[idx] = torch.tensor([0])
                            module.weight = torch.nn.parameter.Parameter(torch.index_select(
                                module.weight, 1, li_in_channels_remain_index[idx]))
                            module.weight = torch.nn.parameter.Parameter(torch.index_select(
                                module.weight, 0, li_out_channels_remain_index[idx]))
                            module.in_channels = len(li_in_channels_remain_index[idx])
                            module.out_channels = len(li_out_channels_remain_index[idx])
                            module.weight = torch.nn.parameter.Parameter(torch.zeros_like(module.weight))  # weight 0
                            continue

                        module.weight = torch.nn.parameter.Parameter(torch.index_select(
                            module.weight, 1, li_in_channels_remain_index[idx]))
                        module.weight = torch.nn.parameter.Parameter(torch.index_select(
                            module.weight, 0, li_out_channels_remain_index[idx]))
                        module.in_channels = len(li_in_channels_remain_index[idx])
                        module.out_channels = len(li_out_channels_remain_index[idx])
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight = torch.nn.parameter.Parameter(torch.index_select(
                        module.weight, 0, li_out_channels_remain_index[idx]))
                    module.bias = torch.nn.parameter.Parameter(
                        torch.index_select(module.bias, 0, li_out_channels_remain_index[idx]))
                    module.running_mean = torch.index_select(
                        module.running_mean, 0, li_out_channels_remain_index[idx])
                    module.running_var = torch.index_select(
                        module.running_var, 0, li_out_channels_remain_index[idx])
                    module.num_features = len(li_out_channels_remain_index[idx])
                    idx += 1
                elif isinstance(module, nn.Linear):
                    module.weight = torch.nn.parameter.Parameter(
                        torch.index_select(module.weight.clone().detach(), 1, li_out_channels_remain_index[-1]))
                    module.in_features = len(li_out_channels_remain_index[-1])


        elif arch == 'resnet50':
            pruned_model = resnet.resnet50()
            pruned_model.load_state_dict(self.model.state_dict(), strict=False)

            idx = 0
            for name, module in pruned_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    if 'downsample' in name:
                        idx -= 1  # follow conv3
                        module.weight = torch.nn.parameter.Parameter(torch.index_select(
                            module.weight, 1, li_in_channels_remain_index[idx-2]))  # follow conv1 in_channels
                        module.weight = torch.nn.parameter.Parameter(torch.index_select(
                            module.weight, 0, li_out_channels_remain_index[idx]))  # follow conv3 out_channels
                        module.in_channels = len(li_in_channels_remain_index[idx-2])
                        module.out_channels = len(li_out_channels_remain_index[idx])
                    else:
                        if len(li_in_channels_remain_index[idx]) == 0:  # pruned all in filters
                            li_in_channels_remain_index[idx] = torch.tensor([0])
                        if len(li_out_channels_remain_index[idx]) == 0:  # pruned all out filters
                            li_out_channels_remain_index[idx] = torch.tensor([0])
                            module.weight = torch.nn.parameter.Parameter(torch.index_select(
                                module.weight, 1, li_in_channels_remain_index[idx]))
                            module.weight = torch.nn.parameter.Parameter(torch.index_select(
                                module.weight, 0, li_out_channels_remain_index[idx]))
                            module.in_channels = len(li_in_channels_remain_index[idx])
                            module.out_channels = len(li_out_channels_remain_index[idx])
                            module.weight = torch.nn.parameter.Parameter(torch.zeros_like(module.weight))  # weight 0
                            continue

                        module.weight = torch.nn.parameter.Parameter(torch.index_select(
                            module.weight, 1, li_in_channels_remain_index[idx]))
                        module.weight = torch.nn.parameter.Parameter(torch.index_select(
                            module.weight, 0, li_out_channels_remain_index[idx]))
                        module.in_channels = len(li_in_channels_remain_index[idx])
                        module.out_channels = len(li_out_channels_remain_index[idx])
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight = torch.nn.parameter.Parameter(torch.index_select(
                        module.weight, 0, li_out_channels_remain_index[idx]))
                    module.bias = torch.nn.parameter.Parameter(
                        torch.index_select(module.bias, 0, li_out_channels_remain_index[idx]))
                    module.running_mean = torch.index_select(
                        module.running_mean, 0, li_out_channels_remain_index[idx])
                    module.running_var = torch.index_select(
                        module.running_var, 0, li_out_channels_remain_index[idx])
                    module.num_features = len(li_out_channels_remain_index[idx])
                    idx += 1
                elif isinstance(module, nn.Linear):
                    module.weight = torch.nn.parameter.Parameter(
                        torch.index_select(module.weight.clone().detach(), 1, li_out_channels_remain_index[-1]))
                    module.in_features = len(li_out_channels_remain_index[-1])

        self.model = pruned_model

    def reconstruction_basic(self):
        flatten_remain_index = []
        for li_li_remain_index in self.remain_index:
            for li_remain_index in li_li_remain_index:
                for remain_index in li_remain_index:
                    flatten_remain_index.append(remain_index)

        idx = 0

        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Conv2d):
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 0, torch.tensor(flatten_remain_index[idx])))
                module.out_channels = len(flatten_remain_index[idx])
                tmp_in_channels = module.out_channels
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 0, torch.tensor(flatten_remain_index[idx])))
                module.bias = torch.nn.parameter.Parameter(torch.index_select(module.bias, 0, torch.tensor(flatten_remain_index[idx])))
                module.running_mean = torch.index_select(module.running_mean, 0, torch.tensor(flatten_remain_index[idx]))
                module.running_var = torch.index_select(module.running_var, 0, torch.tensor(flatten_remain_index[idx]))
                module.num_features = len(flatten_remain_index[idx])

            elif isinstance(module, torch.nn.Sequential):
                for name_, module_ in module.named_children():
                    if isinstance(module_, resnet.BasicBlock):
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                module__.in_channels = tmp_in_channels
                                module__.weight = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.weight, 1, torch.tensor(flatten_remain_index[idx])))
                                idx += 1
                                module__.weight = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.weight, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.out_channels = len(flatten_remain_index[idx])
                                tmp_in_channels = module__.out_channels
                            elif isinstance(module__, torch.nn.BatchNorm2d):  # reconstruct batchnorm considering conv
                                module__.weight = torch.nn.parameter.Parameter(
                                   torch.index_select(module__.weight, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.bias = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.bias, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.running_mean = torch.index_select(module__.running_mean, 0, torch.tensor(flatten_remain_index[idx]))
                                module__.running_var = torch.index_select(module__.running_var, 0, torch.tensor(flatten_remain_index[idx]))
                                module__.num_features = len(flatten_remain_index[idx])
                            elif isinstance(module__, torch.nn.Sequential):  # downsample
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        module___.in_channels = len(flatten_remain_index[idx-2])
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 1,
                                                               torch.tensor(flatten_remain_index[idx-2])))
                                        idx += 1
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.out_channels = len(flatten_remain_index[idx])
                                        tmp_in_channels = module___.out_channels
                                    elif isinstance(module___, torch.nn.BatchNorm2d):
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.bias = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.bias, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.running_mean = torch.index_select(module___.running_mean, 0,
                                                               torch.tensor(flatten_remain_index[idx]))
                                        module___.running_var = torch.index_select(module___.running_var, 0,
                                                               torch.tensor(flatten_remain_index[idx]))
                                        module___.num_features = len(flatten_remain_index[idx])
            elif isinstance(module, torch.nn.Linear):
                module.in_features = tmp_in_channels
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 1, torch.tensor(flatten_remain_index[idx])))

    def reconstruction_bottle(self):
        flatten_remain_index = []
        for li_li_remain_index in self.remain_index:
            if len(li_li_remain_index) == 1:
                flatten_remain_index.append(li_li_remain_index[0])
            else:
                for li_remain_index in li_li_remain_index:
                    for remain_index in li_remain_index:
                        flatten_remain_index.append(remain_index)

        idx = 0

        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Conv2d):
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 0, torch.tensor(flatten_remain_index[idx])))
                module.out_channels = len(flatten_remain_index[idx])
                tmp_in_channels = module.out_channels
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 0, torch.tensor(flatten_remain_index[idx])))
                module.bias = torch.nn.parameter.Parameter(torch.index_select(module.bias, 0, torch.tensor(flatten_remain_index[idx])))
                module.running_mean = torch.index_select(module.running_mean, 0, torch.tensor(flatten_remain_index[idx]))
                module.running_var = torch.index_select(module.running_var, 0, torch.tensor(flatten_remain_index[idx]))
                module.num_features = len(flatten_remain_index[idx])

            elif isinstance(module, torch.nn.Sequential):
                for name_, module_ in module.named_children():
                    if isinstance(module_, resnet.Bottleneck):
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                module__.in_channels = tmp_in_channels
                                module__.weight = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.weight, 1, torch.tensor(flatten_remain_index[idx])))
                                idx += 1
                                module__.weight = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.weight, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.out_channels = len(flatten_remain_index[idx])
                                tmp_in_channels = module__.out_channels
                            elif isinstance(module__, torch.nn.BatchNorm2d):
                                module__.weight = torch.nn.parameter.Parameter(
                                   torch.index_select(module__.weight, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.bias = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.bias, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.running_mean = torch.index_select(module__.running_mean, 0, torch.tensor(flatten_remain_index[idx]))
                                module__.running_var = torch.index_select(module__.running_var, 0, torch.tensor(flatten_remain_index[idx]))
                                module__.num_features = len(flatten_remain_index[idx])
                            elif isinstance(module__, torch.nn.Sequential):  # downsample
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        module___.in_channels = len(flatten_remain_index[idx-3])
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 1,
                                                               torch.tensor(flatten_remain_index[idx-3])))
                                        idx += 1
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.out_channels = len(flatten_remain_index[idx])
                                        tmp_in_channels = module___.out_channels
                                    elif isinstance(module___, torch.nn.BatchNorm2d):
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.bias = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.bias, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.running_mean = torch.index_select(module___.running_mean, 0,
                                                               torch.tensor(flatten_remain_index[idx]))
                                        module___.running_var = torch.index_select(module___.running_var, 0,
                                                               torch.tensor(flatten_remain_index[idx]))
                                        module___.num_features = len(flatten_remain_index[idx])
            elif isinstance(module, torch.nn.Linear):
                module.in_features = tmp_in_channels
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 1, torch.tensor(flatten_remain_index[idx])))

    def reconstruction_model(self, block, SPM, arch):
        assert block in ['basic', 'bottle']
        if SPM == False:
            self.get_remain_index()
            if block == 'basic':
                self.reconstruction_basic()
            elif block == 'bottle':
                self.reconstruction_bottle()
        else:
            self.get_remain_index_SPM(block=block)
            self.reconsruction_bottle_SPM(arch=arch)
        return self.model

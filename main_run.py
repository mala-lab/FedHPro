# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11

from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from utils.long_tailed_setting import train_long_tail
from utils.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, get_class_num, TensorDataset
from utils.sample_dirichlet import clients_indices
from utils.gradient_matching import match_loss, match_meta
from utils.param_aug import DiffAugment
import numpy as np
from torch import max, eq, no_grad, unsqueeze
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from backbone.ResNet10 import resnet10
from tqdm import tqdm
import copy
import torch
import random
import torch.nn.functional as F
import time
import os


def dataset_on_cuda(dataset):
    for attr in ('images', 'labels', 'data', 'targets', 'tensors'):
        value = getattr(dataset, attr, None)
        if torch.is_tensor(value) and value.is_cuda:
            return True
        if isinstance(value, (list, tuple)) and any(torch.is_tensor(item) and item.is_cuda for item in value):
            return True
    return False


def build_dataloader(dataset, batch_size, shuffle=False, device='cpu'):
    use_cuda = str(device).startswith('cuda')
    data_is_cuda = dataset_on_cuda(dataset)
    num_workers = min(6, os.cpu_count() or 0) if use_cuda and not data_is_cuda else 0
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda and not data_is_cuda,
        persistent_workers=num_workers > 0,
    )


def move_batch_to_device(images, labels, device):
    return (
        images.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


class Global(object):
    def __init__(self,
                 num_classes: int,
                 device: str,
                 args,
                 length_of_hp):
        self.device = device
        self.num_classes = num_classes
        self.dim = args.dim
        self.meta_length = args.meta_length
        self.length_of_hp = length_of_hp
        self.rounds_of_hp = args.rounds_of_hp
        self.hyper_hp = torch.randn(size=(args.num_classes * self.length_of_hp, self.dim), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.meta_hp = torch.randn(size=(args.num_classes * self.meta_length, self.dim), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.label_syn = torch.arange(args.num_classes, dtype=torch.long, device=args.device).repeat_interleave(
            self.length_of_hp
        )
        self.label_syn_meta = torch.arange(args.num_classes, dtype=torch.long, device=args.device).repeat_interleave(self.meta_length)
        self.optimizer_hp = SGD([self.hyper_hp,], lr=args.lr_hp)
        self.optimizer_meta = SGD([self.meta_hp,], lr=args.lr_hp)
        self.criterion = CrossEntropyLoss().to(args.device)
        self.meta_net = torch.nn.Linear(self.dim, args.num_classes).to(args.device)
        self.syn_model = resnet10(nclasses=args.num_classes).to(device)


    def obtain_hyper_prototypes(self, args, list_clients_gradient):
        self.syn_model.eval()
        global_cls = self.syn_model.classifier 

        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        for gradient_one in list_clients_gradient: # c*256
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        # gw_real_all each item is [client0, 1, 2, ...] & average gradient
        gw_real_avg = {class_index: [] for class_index in range(args.num_classes)}

        # aggregate 
        for i in range(args.num_classes):
            list_one_class_client_gradient = gw_real_all[i]
            if len(list_one_class_client_gradient) > 0:
                gradients_stack = torch.stack(list_one_class_client_gradient, dim=0) 
                gw_real_avg[i] = gradients_stack.mean(dim=0) # 256
            else:
                gw_real_avg[i] = None

        for _ in range(self.rounds_of_hp):
            loss_virtual = torch.tensor(0.0).to(args.device)
            for c in range(args.num_classes):
                if gw_real_avg[c] is not None:
                    hyper_hp = self.hyper_hp[c * self.length_of_hp:(c + 1) * self.length_of_hp].reshape(
                        (self.length_of_hp, self.dim))
                    lab_syn = self.label_syn[c * self.length_of_hp:(c + 1) * self.length_of_hp]
                    output_syn = global_cls(hyper_hp)
                    loss_syn = self.criterion(output_syn, lab_syn)
                    # compute the HP gradients 
                    gw_syn = torch.autograd.grad(loss_syn, hyper_hp, create_graph=True)[0]
                    loss_virtual += match_loss(gw_syn, gw_real_avg[c], args)
            self.optimizer_hp.zero_grad(set_to_none=True)
            loss_virtual.backward()
            self.optimizer_hp.step()


    def get_agg_params(self, list_dicts_local_params: list, list_nums_local_data: list):
        aggregated_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            aggregated_global_params[name_param] = value_global_param
        return aggregated_global_params


    def meta_revised(self, agg_params):
        hyper_ft = self.meta_hp.detach().clone()
        label_s = self.label_syn_meta.detach().clone()
        dst_load = TensorDataset(hyper_ft, label_s)
        meta_ft = torch.nn.Linear(self.dim, self.num_classes).to(self.device)
        meta_opt = SGD(meta_ft.parameters(), lr=0.01) 
        data_l = build_dataloader(dst_load, batch_size=50, shuffle=True, device=self.device)
        meta_ft.train()
        for _ in range(100):
            for d_b in data_l:
                images, labels = d_b
                images, labels = move_batch_to_device(images, labels, self.device)
                outputs = meta_ft(images)
                loss_m = self.criterion(outputs, labels)
                meta_opt.zero_grad(set_to_none=True)
                loss_m.backward()
                meta_opt.step()
        meta_ft.eval()
        m_params = meta_ft.state_dict()
        for name_param in reversed(agg_params):
            if name_param == 'classifier.bias':
                agg_params[name_param] = m_params['bias']
            if name_param == 'classifier.weight':
                agg_params[name_param] = m_params['weight']
                break

        return copy.deepcopy(meta_ft.state_dict()), copy.deepcopy(agg_params)


    def global_eval(self, data_test, batch_size_test):
        self.syn_model.eval()
        with no_grad():
            test_loader = build_dataloader(data_test, batch_size=batch_size_test, device=self.device)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = move_batch_to_device(images, labels, self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy
    
    def set_meta_params(self, args, global_meta_p, list_meta_g):
        meta_net_params = self.meta_net.state_dict()
        for name_param in reversed(global_meta_p):
            if name_param == 'classifier.bias':
                meta_net_params['bias'] = global_meta_p[name_param]
            if name_param == 'classifier.weight':
                meta_net_params['weight'] = global_meta_p[name_param]
                break
        self.meta_net.load_state_dict(meta_net_params)
        self.meta_net.train()
        all_params = list(self.meta_net.parameters())
        all_meta = {class_index:[] for class_index in range(self.num_classes)}
        for meta_one in list_meta_g:
            for c_num, c_meta in meta_one.items():
                all_meta[c_num].append(c_meta)
        meta_avg = {class_index:[] for class_index in range(self.num_classes)}
        for i in range(self.num_classes):
            meta_temp = []
            one_meta_c = all_meta[i]
            if len(one_meta_c) != 0:
                weight_temp = 1.0 / len(one_meta_c)
                for n_p in range(2):
                    list_vp = []
                    for one_gv in one_meta_c:
                        list_vp.append(one_gv[n_p] * weight_temp)
                    value_gp = sum(list_vp)
                    meta_temp.append(value_gp)
                meta_avg[i] = meta_temp
        for _ in range(100):
            loss_fv = torch.tensor(0.0).to(args.device)
            for c in range(self.num_classes):
                if len(meta_avg[c]) != 0:
                    meta_f = self.meta_hp[c*self.meta_length : (c+1)*self.meta_length].reshape((self.meta_length, self.dim))
                    label_s = self.label_syn_meta[c*self.meta_length : (c+1)*self.meta_length]
                    output_s = self.meta_net(meta_f)
                    loss_meta = self.criterion(output_s, label_s)
                    new_meta_s = torch.autograd.grad(loss_meta, all_params, create_graph=True)
                    loss_fv += match_meta(new_meta_s, meta_avg[c], args)
            self.optimizer_meta.zero_grad(set_to_none=True)
            loss_fv.backward()
            self.optimizer_meta.step()


    def download_params(self):
        return self.syn_model.state_dict()
    

    def revised_params(self, global_p, meta_p):
        for name_p in reversed(global_p):
            if name_p == 'classifier.bias':
                global_p[name_p] = meta_p['bias']
            if name_p == 'classifier.weight':
                global_p[name_p] = meta_p['weight']
                break
        return global_p

    def get_hp(self):
        global_hp = self.hyper_hp.detach().clone()
        return global_hp


class Local(object):
    def __init__(self,
                 data_client,
                 class_list: int,
                 hyper_prototypes,
                 args=None):
        if args is None:
            args = args_parser()
        self.args = args
        self.device = args.device
        self.criterion = CrossEntropyLoss().to(args.device)
        self.repeat_num = args.repeat_num
        self.k_margin = 0.0
        self.local_model = resnet10(nclasses=args.num_classes).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)
        self.hp_info_t = args.hp_info_t
        self.set_client(data_client, class_list, hyper_prototypes)


    def set_client(self, data_client, class_list, hyper_prototypes):
        self.data_client = data_client
        self.class_compose = class_list
        self.hp = {class_index: [] for class_index in range(self.args.num_classes)}
        for c in range(self.args.num_classes):
            self.hp[c] = hyper_prototypes[c * self.args.length_of_hp:(c + 1) * self.args.length_of_hp].reshape(
                        (self.args.length_of_hp, self.args.dim))


    @torch.no_grad()
    def compute_margin(self):
        if self.args.use_margin:
            client_sum = torch.zeros(self.args.num_classes, self.args.dim, device=self.device)
            client_count = torch.zeros(self.args.num_classes, device=self.device)
            lp_loader = build_dataloader(self.data_client, batch_size=100, device=self.device)
            for data_batch in lp_loader:
                images, labels = data_batch
                images, labels = move_batch_to_device(images, labels, self.device)
                features, _ = self.local_model(images)
                client_sum.index_add_(0, labels, features)
                client_count.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
            valid_class = client_count > 0
            lp_stack = client_sum[valid_class] / client_count[valid_class].unsqueeze(1)
            pairwise_products = torch.matmul(lp_stack, lp_stack.T)
            pairwise_no_diag = pairwise_products.clone()
            pairwise_no_diag.fill_diagonal_(0)
            result_tensor = pairwise_no_diag / ((self.args.num_classes - 1) ** 2)
            self.k_margin = torch.sum(result_tensor)


    def compute_gradient(self, args, global_meta_params):
        list_class, per_class_compose = get_class_num(self.class_compose)  # class
        images_all = [unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        labels_all = [self.data_client[i][1] for i in range(len(self.data_client))]
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        indices_class = {
            class_index: (labels_all == class_index).nonzero(as_tuple=True)[0]
            for class_index in list_class
        }
        def get_images(c, n): 
            class_indices = indices_class[c]
            idx_shuffle = torch.randperm(class_indices.numel(), device=args.device)[:n]
            return images_all[class_indices[idx_shuffle]]
        self.local_model.load_state_dict(global_meta_params)
        self.local_model.eval()
        criterion = CrossEntropyLoss().to(args.device)
        self.local_model.classifier.train()
        cls_net_p = list(self.local_model.classifier.parameters())
        # gradients 
        truth_gradient_all = {index: [] for index in list_class}
        truth_meta_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}
        truth_meta_avg = {index: [] for index in list_class}
        # random choose to repeat
        for _ in range(self.repeat_num):
            batch_images = []
            batch_ranges = []
            start = 0
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, args.batch_real)
                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                end = start + img_real.shape[0]
                batch_images.append(img_real)
                batch_ranges.append((c, start, end))
                start = end
            with torch.no_grad():
                feature_batch = self.local_model.features(torch.cat(batch_images, dim=0))
            for c, start, end in batch_ranges:
                lab_real = torch.full((end - start,), c, device=args.device, dtype=torch.long)
                feature_real = feature_batch[start:end]
                feature_real = feature_real.detach().requires_grad_(True)
                output_real = self.local_model.classifier(feature_real)
                loss_real = criterion(output_real, lab_real)
                # compute
                grads = torch.autograd.grad(loss_real, [feature_real] + cls_net_p)
                gw_real = grads[0].detach().clone()
                gw_meta = [g.detach().clone() for g in grads[1:]]
                # print(gw_real[0].shape, gw_meta[0].shape) 
                truth_gradient_all[c].append(gw_real)
                truth_meta_all[c].append(gw_meta)
        for i in list_class:
            gradient_all = truth_gradient_all[i] # list
            gradient_stack = torch.stack(gradient_all, dim=0)
            gw_real_avg = gradient_stack.mean(dim=(0, 1))
            truth_gradient_avg[i] = gw_real_avg # 256
            meta_all = truth_meta_all[i]
            meta_temp = []
            m_weight = 1.0 / len(meta_all)
            for name_param in range(len(meta_all[0])):
                list_values_param = []
                for c_one in meta_all:
                    list_values_param.append(c_one[name_param] * m_weight)
                value_global_param = sum(list_values_param)
                meta_temp.append(value_global_param)
            truth_meta_avg[i] = meta_temp
        
        return truth_gradient_avg, truth_meta_avg
    

    def calculate_infonce(self, f_now, pos_hp, neg_hp, k_margin):
        f_pos = pos_hp.to(self.device)
        f_neg = neg_hp.to(self.device)
        f_combin_hp = torch.cat((f_pos.unsqueeze(0), f_neg), dim=0) 
        f_now_exp = f_now.unsqueeze(0)
        f_now_exp = f_now_exp.expand(f_combin_hp.size(0), f_combin_hp.size(1), -1) 
        cos_sim = F.cosine_similarity(f_now_exp, f_combin_hp, dim=2)
        l = cos_sim.mean(dim=1) 
        l = l / self.hp_info_t
        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1) 
        pos_mask = [1 for _ in range(1)] + [0 for _ in range(f_combin_hp.size(0)-1)]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1) + k_margin 
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss


    def calculate_infonce_batch(self, features, labels, hp_class_norm_mean):
        if self.args.use_margin:
            self.compute_margin()
        features_norm = F.normalize(features, dim=1)
        cu_b_size = features.size(0)
        logits = features_norm.matmul(hp_class_norm_mean.T) / self.hp_info_t
        pos_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        log_den = torch.logsumexp(logits, dim=1)
        if self.k_margin != 0.0:
            log_margin = torch.as_tensor(self.k_margin, device=features.device, dtype=logits.dtype).log()
            log_den = torch.logaddexp(log_den, log_margin)
        return (log_den - pos_logits).mean() / cu_b_size


    def local_train(self, args, global_params):
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip()]
        )
        self.local_model.load_state_dict(global_params)
        self.optimizer.state.clear()
        self.local_model.train()
        hp_stack = torch.stack([self.hp[c] for c in range(self.args.num_classes)], dim=0).to(self.device)
        hp_class_norm_mean = F.normalize(hp_stack, dim=2).mean(dim=1)
        hp_avg = hp_stack.mean(dim=1)
        data_loader = build_dataloader(
            self.data_client,
            batch_size=args.batch_size_local_training,
            shuffle=True,
            device=self.device,
        )
        for _ in range(args.num_epochs_local_training):
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = move_batch_to_device(images, labels, self.device)
                images = transform_train(images)
                features, outputs = self.local_model(images)
                loss_ce = self.criterion(outputs, labels)
                loss_hpcl = self.calculate_infonce_batch(features, labels, hp_class_norm_mean)
                fea_norm = F.normalize(features, dim=1)
                hpavg_norm = F.normalize(hp_avg[labels], dim=1)
                loss_hpal = F.smooth_l1_loss(fea_norm, hpavg_norm, reduction='mean')
                loss = loss_ce + loss_hpcl + loss_hpal 
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return self.local_model.state_dict()


def FedHPro():
    args = args_parser()
    print(
        'imbalance_factor: {ib}, non_iid_factor: {non_iid}\n'
        'length_of_hp: {length_of_hp}, rounds_of_hp: {rounds_of_hp}\n'.format(
        ib=args.imb_factor,
        non_iid=args.non_iid_alpha,
        length_of_hp=args.length_of_hp,
        rounds_of_hp=args.rounds_of_hp)
    )
    random_state = np.random.RandomState(args.seed)
    # load dataset
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    if args.dataset == 'cifar10':
        path_dir = './data/CIFAR10/'
        data_local_training = datasets.CIFAR10(path_dir, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR10(path_dir, train=False, transform=transform_all)
    elif args.dataset == 'cifar100':
        path_dir = './data/CIFAR100/'
        data_local_training = datasets.CIFAR100(path_dir, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR100(path_dir, train=False, transform=transform_all)
    # distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)

    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args,
                          length_of_hp=args.length_of_hp)

    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)
    meta_net = torch.nn.Linear(args.dim, args.num_classes).to(args.device)
    meta_params = meta_net.state_dict()

    # '''training'''
    trained_acc = []
    local_model = None
    for r in tqdm(range(1, args.num_rounds+1), desc='FedHPro-training'):
        global_params = global_model.download_params()
        global_hp = global_model.get_hp()
        global_meta_params = global_model.revised_params(copy.deepcopy(global_params), meta_params)
        # print(global_hp.shape)
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_clients_gradient = []
        list_meta_gradient = []
        list_dicts_local_params = []
        list_nums_local_data = []
        # client
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            if local_model is None:
                local_model = Local(data_client=data_client,
                                    class_list=original_dict_per_client[client],
                                    hyper_prototypes=global_hp,
                                    args=args)
            else:
                local_model.set_client(data_client=data_client,
                                       class_list=original_dict_per_client[client],
                                       hyper_prototypes=global_hp)
            # compute
            truth_gradient, truth_meta = local_model.compute_gradient(args, global_meta_params)
            list_clients_gradient.append(truth_gradient)
            list_meta_gradient.append(truth_meta)
            # update
            local_params = local_model.local_train(args, global_params)
            list_dicts_local_params.append(local_params)
        # aggregating
        aggregated_params = global_model.get_agg_params(list_dicts_local_params, list_nums_local_data)
        global_model.set_meta_params(args, global_meta_params, list_meta_gradient)
        meta_params, agg_params = global_model.meta_revised(aggregated_params)
        global_model.syn_model.load_state_dict(agg_params)
        # updating
        global_model.obtain_hyper_prototypes(args, list_clients_gradient)
        # eval
        one_train_acc = global_model.global_eval(data_global_test, args.batch_size_test)
        # print(one_train_acc)
        trained_acc.append(one_train_acc)
        # if r % 10 == 0:
        print('\t ACC:', trained_acc)
    print('\n ALL_ACC:', trained_acc)


if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    FedHPro()
# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm

from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12

from models.classification_heads import ClassificationHead

from utils import pprint, set_gpu, Timer, count_accuracy, log

import numpy as np
import os
import random


os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network)
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if opt.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'SubspaceTrans':
        cls_head = ClassificationHead(base_learner='SubspaceTrans').cuda()
    elif options.head == 'Subspace':
        cls_head = ClassificationHead(base_learner='Subspace').cuda()
    elif options.head == 'SubspaceFast':
        cls_head = ClassificationHead(base_learner='SubspaceFast').cuda()
    elif opt.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif opt.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif opt.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the classification head type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_test, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--load',
                        default='./save/subspace/best_model.pth', ## your best model
                        help='path of the checkpoint file')
    parser.add_argument('--episode', type=int, default=1000,
                            help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=5,
                            help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='Subspace',
                            help='choose which embedding network to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')

    set_seed(100)

    opt = parser.parse_args()
    (dataset_test, data_loader) = get_dataset(opt)

    set_gpu(opt.gpu)
    
    log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
    log(log_file_path, str(vars(opt)))

    # Define the models
    (embedding_net, cls_head) = get_model(opt)
    
    # Load saved model checkpoints
    saved_models = torch.load(opt.load)
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()
    
    # Evaluate on test set

    # for epp in range(5000):
    #     set_seed(epp)

    aug=True

    test_accuracies = []
    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot,  # num training examples per novel category
        nTestNovel=opt.query * opt.way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode,  # num of batches per epoch
    )

    #print("epp:  ", epp)
    for i, batch in enumerate(tqdm(dloader_test()), 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query

        if opt.shot == 1 and aug:
            flipped_data_support = flip(data_support, 3)
            data_support = torch.cat((data_support, flipped_data_support), dim=0)
            labels_support = torch.cat((labels_support, labels_support), dim=0)
        emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))

        if opt.shot == 1 and aug:
            emb_support = emb_support.reshape(1, n_support*2, -1)
        else:
            emb_support = emb_support.reshape(1, n_support, -1)

        emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
        emb_query = emb_query.reshape(1, n_query, -1)

        if opt.shot == 1 and aug:
            logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot*2)
        else:
            logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)

        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())

        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        if i % 10 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                  .format(i, opt.episode, avg, ci95, acc))

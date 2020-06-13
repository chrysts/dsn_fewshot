import argparse
import os.path as osp
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datamanager.miniimagenet_aug import MiniImageNet
from samplers import CategoriesSampler
from convnet import ConvNet
from algorithm.subspace_projection import Subspace_Projection
from utils import pprint, set_gpu, Averager, Timer, count_acc, flip





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--save-epoch', type=int, default=100)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/subspace-5w5sdiscriminative')
    parser.add_argument('--data-path', default='your miniimagenet folder')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lamb', type=float, default=0.03)

    args = parser.parse_args()
    args.subspace_dim = args.shot-1
    #args.lamb = 0.03
    pprint(vars(args))

    set_gpu(args.gpu)

    trainset = MiniImageNet('train', args.data_path)
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    valset = MiniImageNet('val', args.data_path)
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model = ConvNet().cuda()
    projection_pro = Subspace_Projection(num_dim=args.subspace_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.shot > 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    if args.shot == 1:
        shot_num = 2
        args.subspace_dim = 1
    else:
        shot_num = args.shot

    def save_model(name):
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            qq = p + args.query * args.train_way
            data_shot, data_query = data[:p], data[p:qq]

            if args.shot == 1:
                data_shot = torch.cat((data_shot, flip(data_shot, 3)), dim=0)

            proto = model(data_shot)
            proto = proto.reshape(shot_num, args.train_way, -1)
            proto = torch.transpose(proto, 0, 1)
            hyperplanes, mu = projection_pro.create_subspace(proto, args.train_way, shot_num)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits, discriminative_loss = projection_pro.projection_metric(model(data_query), hyperplanes, mu=mu)
            loss = F.cross_entropy(logits, label) + args.lamb*discriminative_loss
            acc = count_acc(logits, label)

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {},   loss={:.4f} acc={:.4f}'
              .format(epoch,     loss.item(), acc))

        tl = tl.item()
        ta = ta.item()

        if epoch < 100 and epoch%10!=0:
            continue
        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            if args.shot == 1:
                data_shot = torch.cat((data_shot, flip(data_shot, 3)), dim=0)

            proto = model(data_shot)
            proto = proto.reshape(shot_num, args.test_way, -1) ## change to two samples num_shot=2 with flipped one if shot=1
            proto = torch.transpose(proto, 0, 1)
            hyperplanes,  mu = projection_pro.create_subspace(proto, args.test_way, shot_num)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits, _ = projection_pro.projection_metric(model(data_query), hyperplanes, mu=mu)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f} maxacc={:.4f}'.format(epoch, vl, va,trlog['max_acc']))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))



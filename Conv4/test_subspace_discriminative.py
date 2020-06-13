import argparse

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
    parser.add_argument('--save-path', default='./save/subspace-5w5sdiscriminative/max-acc.pth')
    parser.add_argument('--data-path', default='/scratch1/sim314/flush1/miniimagenet/ctm_images/')
    parser.add_argument('--gpu', default='0')
    #parser.add_argument('--subspace-dim', type=int, default=4)
    parser.add_argument('--lamb', type=float, default=5)

    args = parser.parse_args()
    args.subspace_dim = args.shot-1
    pprint(vars(args))

    set_gpu(args.gpu)

    testset = MiniImageNet('test', args.data_path)
    test_sampler = CategoriesSampler(testset.label, 600,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=8, pin_memory=True)

    model = ConvNet().cuda()
    model.load_state_dict(torch.load(args.save_path))
    projection_pro = Subspace_Projection(num_dim=args.subspace_dim)

    if args.shot == 1:
        shot_num = 2
        args.subspace_dim = 1
    else:
        shot_num = args.shot


    trlog = {}
    trlog['test_loss'] = []
    trlog['test_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()


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
    print(' TEST loss={:.4f} acc={:.4f} maxacc={:.4f}'.format( vl, va, trlog['max_acc']))


    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)



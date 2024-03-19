import math
import os.path
import pickle
import numpy as np
import torch
import torch.nn as nn
import random
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib

def calc_recalls(S):
    """
    Computes recall at 1, 5, and 10 given a similarity matrix S.
    By convention, rows of S are assumed to correspond to images and columns are captions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    if isinstance(S, torch.autograd.Variable):
        S = S.data
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

def compute_pooldot_similarity_matrix(image_outputs, audio_outputs, nframes):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    S[i][j] is computed as the dot product between the meanpooled embeddings of
    the ith image output and jth audio output
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 4)
    n = image_outputs.size(0)
    imagePoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    pooled_image_outputs = imagePoolfunc(image_outputs).squeeze(3).squeeze(2)
    audioPoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    pooled_audio_outputs_list = []
    for idx in range(n):
        nF = max(1, nframes[idx])
        pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
    pooled_audio_outputs = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
    S = torch.mm(pooled_image_outputs, pooled_audio_outputs.t())
    return S

def one_imposter_index(i, N):
    imp_ind = random.randint(0, N - 2)
    if imp_ind == i:
        imp_ind = N - 1
    return imp_ind

def basic_get_imposter_indices(N):
    imposter_idc = []
    for i in range(N):
        # Select an imposter index for example i:
        imp_ind = one_imposter_index(i, N)
        imposter_idc.append(imp_ind)
    return imposter_idc

def semihardneg_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S as an autograd.Variable
    Output: The one-way triplet loss from rows of S to columns of S. Impostors are taken
    to be the most similar point to the anchor that is still less similar to the anchor
    than the positive example.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    loss = torch.autograd.Variable(torch.zeros(1).type(S.data.type()), requires_grad=True)
    # Imposter - ground truth
    Sdiff = S - torch.diag(S).view(-1, 1)
    eps = 1e-12
    # All examples less similar than ground truth
    mask = (Sdiff < -eps).type(torch.LongTensor)
    maskf = mask.type_as(S)
    # Mask out all examples >= gt with minimum similarity
    Sp = maskf * Sdiff + (1 - maskf) * torch.min(Sdiff).detach()
    # Find the index maximum similar of the remaining
    _, idc = Sp.max(dim=1)
    idc = idc.data.cpu()
    # Vector mask: 1 iff there exists an example < gt
    has_neg = (mask.sum(dim=1) > 0).data.type(torch.LongTensor)
    # Random imposter indices
    random_imp_ind = torch.LongTensor(basic_get_imposter_indices(N))
    # Use hardneg if there exists an example < gt, otherwise use random imposter
    imp_idc = has_neg * idc + (1 - has_neg) * random_imp_ind
    # This could probably be vectorized too, but I haven't.
    for i, imp in enumerate(imp_idc):
        local_loss = Sdiff[i, imp] + margin
        if (local_loss.data > 0).all():
            loss = loss + local_loss
    loss = loss / N
    return loss

def sampled_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S as an autograd.Variable
    Output: The one-way triplet loss from rows of S to columns of S. Imposters are
    randomly sampled from the columns of S.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    loss = torch.autograd.Variable(torch.zeros(1).type(S.data.type()), requires_grad=True)
    # Imposter - ground truth
    Sdiff = S - torch.diag(S).view(-1, 1)
    imp_ind = torch.LongTensor(basic_get_imposter_indices(N))
    # This could probably be vectorized too, but I haven't.
    for i, imp in enumerate(imp_ind):
        local_loss = Sdiff[i, imp] + margin
        if (local_loss.data > 0).all():
            loss = loss + local_loss
    loss = loss / N
    return loss

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

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    print('now learning rate changed to {:f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate2(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        print('current learing rate is {:f}'.format(lr))
    lr = cur_lr  * 0.1
    print('now learning rate changed to {:f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        target = torch.argmax(target, dim=1)
        target = target.to(output.device)

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def attn_plot(attn, save_path):
    attn_map = attn
    # attn_map = scale_map.detach().mean(0)

    fig, ax = plt.subplots()
    # 绘制热力图
    # norm = matplotlib.colors.Normalize(vmin=0,vmax=0.025)
    # heatmap = ax.imshow(attn_map.cpu().numpy(), cmap='hot', interpolation='nearest', norm=norm)
    heatmap = ax.imshow(attn_map.cpu().numpy(), cmap='hot', interpolation='nearest')

    # 添加颜色条
    cbar = plt.colorbar(heatmap)
    # cbar.set_clim(0, 0.025)

    # 设置横轴和纵轴的标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # 显示图像
    # plt.show()
    plt.savefig(save_path)



PrenetConfig = namedtuple(
  'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])

RNNConfig = namedtuple(
  'RNNConfig',
  ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])

def data_stas():
    data_bar = tqdm(TTA_loader)
    A_labels = []
    for i, (a_input, v_input, labels) in enumerate(data_bar):
        A_labels.extend(torch.argmax(labels, dim=1).numpy())
    print(len(A_labels))
    class_counts = {}
    for label in A_labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    for class_name, count in class_counts.items():
        print(f"Class {class_name}: {count} instances")

    import matplotlib.pyplot as plt

    # 绘制每个类别的实例数量柱状图
    class_indices = list(range(308))
    class_counts = [class_counts[i] for i in class_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(class_indices, class_counts)
    plt.xticks(class_indices, range(308), rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Number of instances')
    plt.title('Number of instances per class in CIFAR10 dataset')
    plt.savefig('/xlearning/mouxing/workspace/MM-TTA/vis_results/stat.png')

    exit()

def plot_gmm(epoch, gmm, X, clean_index, noisy_index, thre, save_path='', density=True):
    plt.clf()
    ax = plt.gca()

    # Compute PDF of whole mixture
    x = np.linspace(0, 1, 1000)
    # logprob = gmm.score_samples(x.reshape(-1, 1))
    # pdf = np.exp(logprob)

    # Compute PDF for each component
    # responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    # pdf_individual = responsibilities * pdf[:, np.newaxis]

    # Plot data histogram
    ax.hist(X[clean_index], bins=100, density=density, histtype='stepfilled', color='green', alpha=0.3,
            label='Clean Samples')
    ax.hist(X[noisy_index], bins=100, density=density, histtype='stepfilled', color='red', alpha=0.3, label='Noisy Samples')

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }
    ax.axvline(thre, color='blue', linestyle='--', label=f'x={thre}')

    # if epoch == 10:
    #     # Plot PDF of whole model
    #     ax.plot(x, pdf, '-k', label='Mixture PDF')
    #
    #     # Plot PDF of each component
    #     ax.plot(x, pdf_individual[:, 0], '--', label='Component A', color='green')
    #     ax.plot(x, pdf_individual[:, 1], '--', label='Component B', color='red')

    # ax.set_xlabel('Per-sample loss, epoch {}'.format(epoch), font1)
    ax.set_xlabel('Per-sample loss', font1)
    ax.set_ylabel('Density', font1)
    x_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=11)

    ax.legend(loc='upper right', prop=font1)
    # if not os.path.exists(save_path[:-7]):
    #     os.mkdir(save_path[:-7])

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss,self).__init__()
        self.gamma = gamma
        self.weight = weight        # 是tensor数据格式的列表

    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        preds = torch.nn.functional.softmax(preds,dim=1)
        # print(preds)
        eps = 1e-7

        target = self.one_hot(preds.size(1), labels)
        # print(target)

        ce = -1 * torch.log(preds+eps) * target
        # print(ce)
        floss = torch.pow((1-preds), self.gamma) * ce
        # print(floss)
        floss = torch.mul(floss, self.weight)
        # print(floss)
        floss = torch.sum(floss, dim=1)
        # print(floss)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0),num))
        one[range(labels.size(0)),labels] = 1
        return one

def plot_scatter(data, save_name, all_acc):
    import matplotlib.pyplot as plt

    # 将数据重新组织成两个列表，一个存储batch_idx，一个存储所有的predicted class
    batch_idx = []
    predicted_classes = []

    for key, values in data.items():
        batch_idx.extend([key] * len(values))
        predicted_classes.extend(values)

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(batch_idx, predicted_classes, color='blue', marker='o', label='Predicted Class', s=6)
    plt.xlabel('batch_idx')
    plt.ylabel('predicted class')
    plt.title('ACC = {}'.format(all_acc))
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
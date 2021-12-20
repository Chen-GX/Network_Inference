import time
import argparse
import pickle
import os
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import random
from torch.autograd import gradcheck
from utils import *
from modules import *

# ============ 训练参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--random', action='store_true', default=True,  # 是否控制随机种子
                    help='Control Random seed.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--lr_decay', type=int, default=2,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--prediction_steps', type=int, default=1, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--step', action='store_true', default=False)
# ============ 模型参数 ==================
parser.add_argument('--encoder', type=str, default='CNN',
                    help='Type of path encoder model (CNN, CNN_Full).')
parser.add_argument('--decoder', type=str, default='GAT',
                    help='Type of decoder model (GAT).')
parser.add_argument('--decoder_hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

# ============ 数据集参数 ==================
parser.add_argument('--num_nodes', type=int, default=16,
                    help='Number of atoms in simulation.')
parser.add_argument('--timesteps', type=int, default=10,
                    help='The number of time steps per sample.')
parser.add_argument('--edge_type', type=int, default=2)

# ============ 文件存储 ==================
parser.add_argument('--save_folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--load_data_log', type=str, default='16000_2000_n16_e32_tt10',
                    help='The cascade data after processing.')

args = parser.parse_args()
args.feat_dims = args.num_nodes
args.cuda = not args.no_cuda and torch.cuda.is_available()

index = []
for _ in range(args.edge_type):
    index.append(_)
args.index = torch.tensor(index, dtype=torch.float32)

# 设置随机种子
if args.random:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    timestamp = timestamp.replace(':', '%')
    timestamp = timestamp.replace('.', '%')
    save_folder = '{}/exp_{}_{}/'.format(args.save_folder, args.load_data_log, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

if args.encoder == 'CNN':
    encoder = CNNEncoder(args.timesteps, args.num_nodes, args.feat_dims)

if args.decoder == 'GAT':
    decoder = GAT(args.feat_dims,
                  nhid=args.decoder_hidden,
                  nclass=args.feat_dims,
                  dropout=args.dropout,
                  nheads=args.nb_heads,
                  alpha=args.alpha)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

# cascde [num_batch, num_timesteps, num_nodes, num_dims]
# network [num_batch, num_nodes, num_nodes]
train_loader, valid_loader, test_loader = load_data(args)
# weight_decay=args.weight_decay
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)
I = torch.eye(args.num_nodes)
if args.cuda:
    encoder.cuda()
    decoder.cuda()
    args.index = args.index.cuda()
    I = I.cuda()


def train(epoch, best_val_loss, encoder, decoder):
    t = time.time()
    loss_train = []
    acc_train = []
    f1_train = []
    pre_train = []
    rec_train = []

    encoder.train()
    decoder.train()
    if args.step:
        print("Epoch{}:".format(epoch))
    for batch_idx, (data, network) in enumerate(train_loader):
        if args.cuda:
            data, network = data.cuda(), network.cuda()

        # data[128, 10, 16, 16]
        optimizer.zero_grad()

        logits = encoder(data)  # logits [128, 16, 16, 2]
        adj = softargmax(logits, args.index)  # A [128, 16, 16]
        pred_net = torch.round(adj.detach())
        adj = adj + I
        output = decoder(data, adj, args.prediction_steps)
        target = data[:, args.prediction_steps:, :, :]
        loss = nll_gaussian_loss(output, target)
        loss.backward()

        optimizer.step()
        acc, f1, pre, rec = edge_accuracy_f1(pred_net, network)
        acc_train.append(acc)
        f1_train.append(f1)
        pre_train.append(pre)
        rec_train.append(rec)
        loss_train.append(loss.item())
        if args.step:
            print('loss_train: {:.10f}'.format(loss.item()),
                  'acc_train: {:.10f}'.format(acc),
                  'f1_train: {:.10f}'.format(f1),
                  'pre_train: {:.10f}'.format(pre),
                  'rec_train: {:.10f}'.format(rec))

    loss_val = []
    acc_val = []
    f1_val = []
    pre_val = []
    rec_val = []

    encoder.eval()
    decoder.eval()
    for batch_idx, (data, network) in enumerate(valid_loader):
        if args.cuda:
            data, network = data.cuda(), network.cuda()

        logits = encoder(data)  # logits [128, 16, 16, 2]
        adj = softargmax(logits, args.index)  # A [128, 16, 16]
        pred_net = torch.round(adj.detach())
        adj = adj + I

        output = decoder(data, adj, args.prediction_steps)
        target = data[:, args.prediction_steps:, :, :]
        loss = nll_gaussian_loss(output, target)

        acc, f1, pre, rec = edge_accuracy_f1(pred_net, network)
        acc_val.append(acc)
        f1_val.append(f1)
        pre_val.append(pre)
        rec_val.append(rec)

        loss_val.append(loss.item())

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.10f}'.format(np.mean(loss_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'f1_train: {:.10f}'.format(np.mean(f1_train)),
          'pre_train: {:.10f}'.format(np.mean(pre_train)),
          'rec_train: {:.10f}'.format(np.mean(rec_train)),
          'loss_val: {:.10f}'.format(np.mean(loss_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)),
          'f1_val: {:.10f}'.format(np.mean(f1_val)),
          'pre_val: {:.10f}'.format(np.mean(pre_val)),
          'rec_val: {:.10f}'.format(np.mean(rec_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(loss_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'f1_train: {:.10f}'.format(np.mean(f1_train)),
              'pre_train: {:.10f}'.format(np.mean(pre_train)),
              'rec_train: {:.10f}'.format(np.mean(rec_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'f1_val: {:.10f}'.format(np.mean(f1_val)),
              'pre_val: {:.10f}'.format(np.mean(pre_val)),
              'rec_val: {:.10f}'.format(np.mean(rec_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(loss_val)


def test():
    acc_test = []
    f1_test = []
    pre_test = []
    rec_test = []
    loss_test = []

    encoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))
    for batch_idx, (data, network) in enumerate(test_loader):
        if args.cuda:
            data, network = data.cuda(), network.cuda()

        logits = encoder(data)  # logits [128, 16, 16, 2]
        adj = softargmax(logits, args.index)  # A [128, 16, 16]
        pred_net = torch.round(adj.detach())
        adj = adj + I

        output = decoder(data, adj, args.prediction_steps)
        target = data[:, args.prediction_steps:, :, :]
        loss = nll_gaussian_loss(output, target)

        acc, f1, pre, rec = edge_accuracy_f1(pred_net, network)
        acc_test.append(acc)
        f1_test.append(f1)
        pre_test.append(pre)
        rec_test.append(rec)

        loss_test.append(loss.item())

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('loss_test: {:.10f}'.format(np.mean(loss_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)),
          'f1_test: {:.10f}'.format(np.mean(f1_test)),
          'pre_test: {:.10f}'.format(np.mean(pre_test)),
          'rec_test: {:.10f}'.format(np.mean(rec_test)))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              'f1_test: {:.10f}'.format(np.mean(f1_test)),
              'pre_test: {:.10f}'.format(np.mean(pre_test)),
              'rec_test: {:.10f}'.format(np.mean(rec_test)),
              file=log)
        log.flush()


# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    epoch_val_loss = train(epoch, best_val_loss, encoder, decoder)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch
    scheduler.step()

print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

test()
if log is not None:
    print(save_folder)
    log.close()

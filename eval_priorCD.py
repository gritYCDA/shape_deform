import os
import time
import argparse
import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from lib.auto_encoder import PointCloudAE
from lib.loss import ChamferLoss
from data.shape_dataset import ShapeDataset
from lib.utils import setup_logger
import numpy as np
import pdb;

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024, help='number of points, needed if use points')
parser.add_argument('--emb_dim', type=int, default=512, help='dimension of latent embedding [default: 512]')
parser.add_argument('--h5_file', type=str, default='data/obj_models/ShapeNetCore_4096.h5', help='h5 file')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=50, help='max number of epochs to train')
parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='results/ae_points', help='directory to save train results')
opt = parser.parse_args()

opt.repeat_epoch = 10
opt.decay_step = 5000
opt.decay_rate = [1.0, 0.6, 0.3, 0.1]
mean_shapes = np.load('assets/mean_points_emb.npy')


def train_net():
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    # tb_writer = tf.summary.FileWriter(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # model & loss
    # estimator = PointCloudAE(opt.emb_dim, opt.num_point)
    # estimator.cuda()
    criterion = ChamferLoss()
    # if opt.resume_model != '':
    #     estimator.load_state_dict(torch.load(opt.resume_model))
    # dataset
    # train: 1102, val: 206
    # train_dataset = ShapeDataset(opt.h5_file, mode='train', augment=True)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
    #                                                shuffle=True, num_workers=opt.num_workers)
    val_dataset = ShapeDataset(opt.h5_file, mode='val', augment=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=opt.num_workers)

    val_loss = 0.0
    cd_num = torch.zeros(6)
    cd = torch.zeros(6)
    for i, data in enumerate(val_dataloader, 1):
        batch_xyz, batch_label = data
        idx = batch_label.item()
        batch_xyz = batch_xyz[:, :, :3].cuda()
        prior = torch.from_numpy(mean_shapes[idx]).cuda().float().unsqueeze(0)
        loss, _, _ = criterion(prior, batch_xyz)
        cd_num[idx] += 1
        cd[idx] += loss.item()

    # zero divider
    cd_metric = (cd / cd_num) * 1000
    print("{:.2f} : {:.2f} : {:.2f} : {:.2f} : {:.2f} : {:.2f} : {:.2f}".format(cd_metric[0], cd_metric[1], cd_metric[2], cd_metric[3], cd_metric[4], cd_metric[5], torch.mean(cd_metric)))

if __name__ == '__main__':
    train_net()
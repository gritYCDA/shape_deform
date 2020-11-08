import os
import time
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.network import DeformNet
from lib.align import estimateSimilarityTransform
from lib.utils import load_depth, get_bbox, compute_mAP, plot_mAP
from lib.loss import ChamferLoss


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='real_test', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--model', type=str, default='results/real/model_50.pth', help='resume from saved model : /camera/model_50.pth, /real/model_50.pth')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
opt = parser.parse_args()

mean_shapes = np.load('assets/mean_points_emb.npy')

assert opt.data in ['val', 'real_test']
if opt.data == 'val':
    result_dir = 'results/eval_camera'
    file_path = 'CAMERA/val_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
else:
    result_dir = 'results/eval_real'
    file_path = 'Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

model_file_path = ['obj_models/real_test.pkl']
models = {}
for path in model_file_path:
    with open(os.path.join(opt.data_dir, path), 'rb') as f:
        models.update(cPickle.load(f))
print('{} models loaded.'.format(len(models)))


def detect():
    # resume model
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    estimator = DeformNet(opt.n_cat, opt.nv_prior)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()
    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data_dir, file_path))]
    # TODO: test, chamfer distance
    chamferD = ChamferLoss()
    cd_num = torch.zeros(6)
    prior_cd = torch.zeros(6)
    deform_cd = torch.zeros(6)
    for path in tqdm(img_list):
        img_path = os.path.join(opt.data_dir, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        raw_depth = load_depth(img_path)
        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join('results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        num_insts = len(mrcnn_result['class_ids'])
        f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
        f_size = np.zeros((num_insts, 3), dtype=float)
        # prepare frame data
        f_points, f_rgb, f_choose, f_catId, f_prior, f_model = [], [], [], [], [], []
        valid_inst = []
        for i in range(num_insts):
            cat_id = mrcnn_result['class_ids'][i] - 1
            prior = mean_shapes[cat_id]
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])
            mask = np.logical_and(mrcnn_result['masks'][:, :, i], raw_depth > 0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            # no depth observation for background in CAMERA dataset
            # beacuase of how we compute the bbox in function get_bbox
            # there might be a chance that no foreground points after cropping the mask
            # cuased by false positive of mask_rcnn, most of the regions are background
            if len(choose) < 32:
                f_sRT[i] = np.identity(4, dtype=float)
                f_size[i] = 2 * np.amax(np.abs(prior), axis=0)
                continue
            else:
                valid_inst.append(i)
            # process objects with valid depth observation
            if len(choose) > opt.n_pts:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:opt.n_pts] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, opt.n_pts-len(choose)), 'wrap')
            depth_masked = raw_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            pt2 = depth_masked / norm_scale
            pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
            points = np.concatenate((pt0, pt1, pt2), axis=1)
            rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
            rgb = cv2.resize(rgb, (opt.img_size, opt.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = norm_color(rgb)
            crop_w = rmax - rmin
            ratio = opt.img_size / crop_w
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio) * opt.img_size + np.floor(col_idx * ratio)).astype(np.int64)
            # concatenate instances
            try:
                idx_gt = np.argwhere(gts['class_ids'] - 1 == cat_id).item()
            except:
                valid_inst.remove(i)
                continue
            model = models[gts['model_list'][idx_gt]].astype(np.float32)  # 1024 points
            f_model.append(model)
            f_points.append(points)
            f_rgb.append(rgb)
            f_choose.append(choose)
            f_catId.append(cat_id)
            f_prior.append(prior)
        if len(valid_inst):
            f_points = torch.cuda.FloatTensor(f_points)
            f_rgb = torch.stack(f_rgb, dim=0).cuda()
            f_choose = torch.cuda.LongTensor(f_choose)
            f_catId = torch.cuda.LongTensor(f_catId)
            f_prior = torch.cuda.FloatTensor(f_prior)
            f_model = torch.cuda.FloatTensor(f_model)
            # inference
            torch.cuda.synchronize()
            assign_mat, deltas = estimator(f_points, f_rgb, f_choose, f_catId, f_prior)
            # assign_mat, deltas = estimator(f_rgb, f_choose, f_catId, f_prior)
            # reconstruction points
            inst_shape = f_prior + deltas.detach()

            for i in range(len(valid_inst)):
                prior_loss, _, _ = chamferD(f_prior[i].unsqueeze(0), f_model[i].unsqueeze(0))
                deform_loss, _, _ = chamferD(inst_shape[i].unsqueeze(0), f_model[i].unsqueeze(0))

                idx = f_catId[i]
                cd_num[idx] += 1
                prior_cd[idx] += prior_loss.item()
                deform_cd[idx] += deform_loss.item()


    deform_cd_metric = (deform_cd / cd_num) * 1000
    print(
        "recon: {:.2f} , {:.2f} , {:.2f} , {:.2f} , {:.2f} , {:.2f} , {:.2f}".format(deform_cd_metric[0], deform_cd_metric[1],
                                                                              deform_cd_metric[2], deform_cd_metric[3],
                                                                              deform_cd_metric[4], deform_cd_metric[5],
                                                                              torch.mean(deform_cd_metric)))
    prior_cd_metric = (prior_cd / cd_num) * 1000
    print("prior: {:.2f} , {:.2f} , {:.2f} , {:.2f} , {:.2f} , {:.2f} , {:.2f}".format(prior_cd_metric[0], prior_cd_metric[1],
                                                                                prior_cd_metric[2], prior_cd_metric[3],
                                                                                prior_cd_metric[4], prior_cd_metric[5],
                                                                                torch.mean(prior_cd_metric)))


if __name__ == '__main__':
    print('Detecting ...')
    detect()


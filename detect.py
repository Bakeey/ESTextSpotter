import os, sys
import torch
import numpy as np

from models.ests import build_ests
from util.slconfig import SLConfig
from util import box_ops
from PIL import Image
import datasets.transforms as T
import datetime
import pickle

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms
import matplotlib.font_manager as mfm
import click

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
def _decode_recognition(rec):
    s = ''
    rec = rec.tolist()
    for c in rec:
        if c>94:
            continue
        s += CTLABELS[c]
    return s

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    args.device = 'cuda'
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


class ColorMap():
    def __init__(self, basergb=[255,255,0]):
        self.basergb = np.array(basergb)
    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1) # h, w, 3
        attn1 = attnmap.copy()[..., None] # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res
    

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)
    
def visualize(img, tgt, caption=None, dpi=300, savedir=None, show_in_console=False):
    """
    img: tensor(3, H, W)
    tgt: make sure they are all on cpu.
        must have items: 'image_id', 'boxes', 'size'
    """
    plt.figure(dpi=dpi)
    plt.rcParams['font.size'] = '5'
    ax = plt.gca()
    img = renorm(img).permute(1, 2, 0)
    # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
    #     import ipdb; ipdb.set_trace()
    ax.imshow(img)
    
    addtgt(tgt)
    # if show_in_console:
    #     plt.show()

    if savedir is not None:
        image_filename = tgt['image']  # Extract the filename from the target dictionary
        if caption is None:
            basename = os.path.splitext(image_filename)[0]  # Strip the original extension
            savename = f"{basename}-{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}.png"
        else:
            savename = '{}/{}-{}-{}.png'.format(savedir, caption, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
        full_path = os.path.join(savedir, savename)
        os.makedirs(savedir, exist_ok=True)  # Ensure the directory exists
        plt.savefig(full_path)
        print(f"Image saved as: {full_path}")
    plt.close()

    if show_in_console:
        plt.show()

    plt.close()

def addtgt(tgt):
    """
    
    """
    assert 'boxes' in tgt
    ax = plt.gca()
    H, W = tgt['size'].tolist() 
    numbox = tgt['boxes'].shape[0]

    color = []
    polygons = []
    boxes = []
    for box, bezier in zip(tgt['boxes'].cpu(), tgt['beziers'].cpu()):
        unnormbbox = box * torch.Tensor([W, H, W, H])
        unnormbbox[:2] -= unnormbbox[2:] / 2
        np_poly = bezier.reshape(-1,2)* torch.Tensor([W, H])
        [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
        boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        # np_poly = np.array(poly).reshape((4,2))
        polygons.append(Polygon(np_poly.numpy()))
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)

    if 'box_label' in tgt:
        assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
        for idx, bl in enumerate(tgt['box_label']):
            _string = str(bl)
            bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
            # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
            ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

    if 'caption' in tgt:
        ax.set_title(tgt['caption'], wrap=True)

    if 'attn' in tgt:
        if isinstance(tgt['attn'], tuple):
            tgt['attn'] = [tgt['attn']]
        for item in tgt['attn']:
            attn_map, basergb = item
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-3)
            attn_map = (attn_map * 255).astype(np.uint8)
            cm = ColorMap(basergb)
            heatmap = cm(attn_map)
            ax.imshow(heatmap)
    ax.set_axis_off()

    
@click.command()
@click.option('--model_config_path', type=str, default="./config/ESTS/ESTS_5scale_ctw1500_finetune.py", help='Path to the model config file')
@click.option('--model_checkpoint_path', type=str, default="/cluster/home/kiten/totaltext_checkpoint.pth", help='Path to the model checkpoint')
@click.option('--image_dir', type=str, default='/cluster/home/kiten/images/test', help='Path to the image directory')
@click.option('--out_dir', type=str, default='/cluster/home/kiten/output/test/text_detections', help='Path to the output directory')
@click.option('--out_dir_vis', type=str, default='/cluster/home/kiten/output/test/images', help='Path to the output directory for visualization')
def main(model_config_path, model_checkpoint_path, image_dir, out_dir, out_dir_vis):

    args = SLConfig.fromfile(model_config_path) 
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    transform = T.Compose([
        T.RandomResize([800],max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    )
    dir = os.listdir(image_dir)
    for idx, filename in enumerate(dir):
        image = Image.open(os.path.join(image_dir, filename)).convert('RGB')
        image, _ = transform(image,None)
        output = model(image[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]
        rec = [_decode_recognition(i) for i in output['rec']]
        threshold = 0.4
        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > threshold
        recs = []
        recs_all = []
        for i,r in zip(select_mask,rec):
            if i:
                recs.append(r)
            recs_all.append(r)
        scores = []
        scores_all = []
        for i,s in zip(select_mask,scores):
            if i:
                scores.append(s)
            scores_all.append(s)
        # box_label = ['text' for item in rec[select_mask]]
        pred_dict = {
            'all_boxes': boxes,
            'all_beziers': output['beziers'],
            'all_recs': recs_all,
            'all_scores': scores_all,
            'boxes': boxes[select_mask],
            'size': torch.tensor([image.shape[1],image.shape[2]]),
            'box_label': recs,
            'box_scores': scores,
            'image_id' : idx,
            'image' : filename,
            'beziers': output['beziers'][select_mask]
        }

        # Move all tensors to CPU before serializing
        for key, value in pred_dict.items():
            if isinstance(value, torch.Tensor):
                pred_dict[key] = value.cpu()
        
        # Replace the current extension with .pkl
        base_name, _ = os.path.splitext(filename)
        with open(os.path.join(out_dir, base_name + '.pkl'), 'wb') as f:
            pickle.dump(pred_dict, f)
        visualize(image, pred_dict, savedir=out_dir_vis)

if __name__ == '__main__':
    main()
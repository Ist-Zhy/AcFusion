# coding:utf-8
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from TaskFusion_dataset import Fusion_dataset
from FusionNetwork import FusionNet
from torch.autograd import Variable
from PIL import Image
from Conversion import YCrCb2RGB, RGB2YCrCb

def main():
    fusion_model_path = 'model/Fusion/fusion_model.pth'
    fusionmodel = eval('FusionNet')(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        fusionmodel.to(device)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    ir_path = 'Test/ir/'
    vi_path = 'Test/vi/'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)

            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AcFusion with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='AcFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    fused_dir = 'Fusion_Result/Model'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()

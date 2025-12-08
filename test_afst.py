import torch
import argparse
import os
import json
from utils.data_loader import DataLoader
from tqdm import tqdm
from utils.util import calc_coefficient
from utils.afst import AFSTMIQA
from modules.CL_model import CONTRIQUE_model
from torchvision.models import resnet50

def test_fn(args):
    root_dir = os.getcwd()
    pmiqd_path = os.path.join(root_dir, 'dataset_miqa', 'pmiqd')

    folder_path = {
        'pmiqd': pmiqd_path,
    }

    img_num = {
        'pmiqd': list(range(1, 425)),
        # 'pmiqd': list(range(1, 6)),  
    }

    print('Testing on <{}> dataset'.format(args.dset.upper()))

    # SEED
    if args.seed == 0:
        pass
    else:
        print('SEED = {}'.format(args.seed))
        import random
        import numpy as np
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    total_num_images = img_num[args.dset]
    # Load test index
    test_idx_path = f'{args.log_path}' + '/' + 'test_idx' + '_' + str(args.seed) + '.json'
    with open(test_idx_path, 'r') as f:
        test_index = json.load(f)

    mos_file = os.path.join(pmiqd_path, 'mos.xlsx')
    import pandas as pd
    df = pd.read_excel(mos_file)
    mos_scores = df.iloc[:, 0].values

    dataloader_test = DataLoader(args.dset, folder_path[args.dset],
                                 test_index, args.psize, args.tnum,
                                 args.bsize, istrain=False).get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = AFSTMIQA(contrast_dim=2048).to(device)
    # Load trained model
    ckpt = torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'), map_location=device)
    model.load_state_dict(ckpt['net'])
    model.eval()

    # encoder
    encoder = resnet50(pretrained=False)
    cmodel = CONTRIQUE_model(args, encoder, 2048).to(device)
    c_ckpt = torch.load(args.model_path_CONTRIQUE, map_location=device)
    cmodel.load_state_dict(c_ckpt)
    cmodel.eval()

    for param in cmodel.parameters():
        param.requires_grad = False

    print(f'+====================+ Testing +====================+')
    sp, pl = calc_coefficient(dataloader_test, model, device, args.pnum, cmodel)
    print(f'SROCC: {sp:.4f}, PLCC: {pl:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dset', type=str, default='pmiqd', help='dataset')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning_rate')
    parser.add_argument('--bsize', type=int, default=8, help='batch_size')
    parser.add_argument('--psize', type=int, default=224, help='patch_size')
    parser.add_argument('--pnum', type=int, default=8, help='train patch_num')
    parser.add_argument('--tnum', type=int, default=8, help='test patch_num')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight_decay')
    parser.add_argument('--tm', type=int, default=50, help='T_max')
    parser.add_argument('--sv_path', type=str, default='sav', help='save_path')
    parser.add_argument('--tb_path', type=str, default='sav/tensorboard', help='tensorboard_path')
    parser.add_argument('--log_path', type=str, default='sav/log', help='log_path')
    parser.add_argument('--model_path', type=str, default='sav/model_m', help='save_model_path')
    parser.add_argument('--model_path_CONTRIQUE', type=str, default='models/CONTRIQUE_checkpoint25.tar',
                        help='Path to trained CONTRIQUE model')
    parser.add_argument('--model_name', type=str, default='afst_miqa', help='model name')

    args = parser.parse_args()

    test_fn(args)
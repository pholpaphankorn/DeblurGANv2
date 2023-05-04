import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeblurGANv2")
    parser.add_argument("--input_folder",default="")
    parser.add_argument("--output_folder",default="")
    parser.add_argument("--weights_path",default="")
    parser.add_argument("--configs_path",default="")




    return parser.parse_args()


class Predictor:
    def __init__(self, weights_path: str, model_name: str = '',configs_path: str=''):

        with open(configs_path,encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)

        model = get_generator(model_name or config['model'])
        #for fpn_inception.h5 only
        if not os.path.exists(weights_path):
            file_id='1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR'
            output_path='../DeblurGANv2/pretrained_weights/fpn_inception.h5'
            os.system(f'wget --load-cookies /tmp/cookies.txt \
                      \"https://docs.google.com/uc?export=download&confirm=$(wget \
                      --quiet \
                      --save-cookies /tmp/cookies.txt \
                      --keep-session-cookies \
                      --no-check-certificate \'https://docs.google.com/uc?export=download&id={file_id}\' \
                      -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p\')&id={file_id}" \
                      -O {output_path} \
                      && rm -rf /tmp/cookies.txt')
        model.load_state_dict(torch.load(weights_path)['model'])

        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)

def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path='fpn_inception.h5',
         out_dir='submit/test_img',
         side_by_side: bool = False,
         video: bool = False,
         configs_path:str=''):
    def sorted_glob(pattern):
        return sorted(glob(pattern))
    

    imgs = sorted_glob(img_pattern)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    predictor = Predictor(weights_path=weights_path,configs_path=configs_path)

    os.makedirs(out_dir, exist_ok=True)
    if not video:
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair
            img, mask = map(cv2.imread, (f_img, f_mask))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = predictor(img, mask)
            if side_by_side:
                pred = np.hstack((img, pred))
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, name),
                        pred)
    else:
        process_video(pairs, predictor, out_dir)

# def getfiles():
#     filenames = os.listdir(r'.\dataset1\blur')
#     print(filenames)
def get_files(input_folder):
    list=[]
    for filepath,dirnames,filenames in os.walk(input_folder):
        for filename in filenames:
            print(f'found {filename}')
            list.append(os.path.join(filepath,filename))
    return list




import ssl


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
  #  Fire(main)
#增加批量处理图片：
    args = parse_args()
    img_path=get_files(args.input_folder)
    for i in img_path:
        main(i,side_by_side=False,out_dir=args.output_folder,weights_path=args.weights_path,configs_path=args.configs_path)
    # main('test_img/tt.mp4')

import pickle
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict

from .basedata import BaseData


class ThreeDFuture(BaseData):
    def __init__(self, cfg, dataset, split, train):
        super().__init__(cfg, dataset, split, train)
        self.root_dir = Path('/mnt/vol_c/data/3d-future')
        self.file_name = 'chairs.pkl'
        self.image_dir = self.root_dir
        self.preload_anno()
        print('3d-future/' + self.file_name, len(self))
    
    def compute_mask_and_bbox(self, rel_path: str):
        image = np.array(Image.open(self.root_dir / rel_path))
        mask = (image[:, :, 3] > 0).astype(int)
        x,y = np.where(mask == 0)
        min_x, min_y = x.min(), y.min()
        max_x, max_y = x.max(), y.max()
        bbox = np.array([min_x, min_y, max_x, max_y])
        return mask, bbox 

    def preload_anno(self):
        file_path = self.root_dir / self.file_name
        full_data = pickle.load(open(file_path, "rb"))
        self.anno['text'] = [] 
        self.anno['pointcloud'] = []
        data = full_data[self.split]
        for d in tqdm.tqdm(data):
            id, view, text, pointcloud = d['id'], d['view'], d['text'], d['pointcloud']
            rel_path = id + f'/colors_{view}.png'
            self.anno['rel_path'].append(rel_path)

            mask, bbox = self.compute_mask_and_bbox(rel_path)
            self.anno['mask'].append(mask)
            self.anno['bbox'].append(bbox)
            
            # Form text descriptions
            # print(text)
            type, style, _, material = text.split(',')
            # print(type, style, material)
            type_vars = type.split('/')

            text = f"{style.strip()} {type_vars[0].lower().strip()} made out of {material.lower().strip()}"
            # print(text)

            self.anno['text'].append(text)
            self.anno['pointcloud'].append(pointcloud)

    def clear_sample(self, sample: Dict):
        out = super().clear_sample(sample)
        out['text'] = sample['text']
        out['pointcloud'] = sample['pointcloud']

        return out

    def get_datapoint(self, index):
        sample = super().get_datapoint(index)
        sample['pointcloud'] = self.anno['pointcloud'][index]
        sample['text'] = self.anno['text'][index]

        if self.know_mean > 0:
            sample['mean_shape'] = self.get_mean_shape(index)

        return sample

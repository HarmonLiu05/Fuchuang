import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class ChimpanzeeDataset(Dataset):
    def __init__(self, root_dir, annotation_file, image_dir, transform=None, 
                 min_samples_per_identity=20, identities=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.transform = transform
        
        self.samples = self._parse_annotations(
            os.path.join(root_dir, annotation_file),
            min_samples_per_identity,
            identities
        )
        
        self.identity_to_idx = {name: i for i, name in enumerate(self.identities)}
        
    def _parse_annotations(self, ann_file, min_samples, specified_identities):
        identity_samples = defaultdict(list)
        
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                filename = parts[0]
                identity = self._extract_identity_from_filename(filename)
                
                if identity:
                    identity_samples[identity].append(filename)
        
        if specified_identities:
            self.identities = specified_identities
        else:
            self.identities = [
                name for name, samples in identity_samples.items()
                if len(samples) >= min_samples
            ]
        
        self.identity_to_idx = {name: i for i, name in enumerate(self.identities)}
        
        samples = []
        for identity in self.identities:
            for filename in identity_samples[identity]:
                samples.append({
                    'filename': filename,
                    'identity': identity,
                    'label': self.identity_to_idx[identity]
                })
        
        return samples
    
    def _extract_identity_from_filename(self, filename):
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        if len(parts) >= 1:
            return parts[0]
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']

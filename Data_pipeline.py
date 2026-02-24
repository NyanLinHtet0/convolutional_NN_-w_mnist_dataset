# Data_pipeline.py
import os
from pathlib import Path
import numpy as np
from PIL import Image
class DataPipeline:
    def __init__(self, root_dir="Data", exts=(".png",)):
        self.root_dir = Path(root_dir)
        self.exts = exts

    def _list_images(self, folder):
        return sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in self.exts]
        )
    
    
    def load_and_npz_save(self, split="train", target_size=None, normalize=True):
        split_path = self.root_dir / split
        class_folders = sorted([p for p in split_path.iterdir() if p.is_dir()],
                               key=lambda p: p.name)
        X = []
        y = []

        for label, folder in enumerate(class_folders):
            images = self._list_images(folder)

            for img_path in images:
                img = Image.open(img_path).convert("L")  # grayscale

                if target_size is not None:
                    img = img.resize(target_size, Image.BILINEAR)

                arr = np.array(img, dtype=np.float32)

                if normalize:
                    arr /= 255.0

                X.append(arr)
                y.append(label)

        X = np.stack(X)   # (N, H, W)
        y = np.array(y)
        
        np.savez(f'{split}_{X.shape[1]}x{X.shape[2]}_dataset.npz', images=X, labels=y)
        return X, y
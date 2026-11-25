"""Training data: an image-folder dataset of ALIGNED 112x112 faces  root/<identity>/*.jpg .

Convention (must match inference, see pipeline/embedder.py): RGB, pixels scaled to [-1, 1].
"""
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class RandomDegrade:
    """Quality augmentation: with prob p apply 1-2 of {low-resolution, Gaussian blur, JPEG} with random strength.

    Training ranges are deliberately milder than the hardest evaluation cases (16px, blur sigma=4, JPEG q=5),
    so the hard-case study still measures generalisation, not just memorised test conditions.
    """

    def __init__(self, p=0.5, size=(24, 112), sigma=(0.5, 3.5), quality=(10, 95)):
        self.p, self.size, self.sigma, self.quality = p, size, sigma, quality

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        a = np.asarray(img)
        for op in sorted(random.sample(("lowres", "blur", "jpeg"), random.choice((1, 2)))):
            if op == "lowres":
                s = random.randint(*self.size)
                a = cv2.resize(cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA), (112, 112),
                               interpolation=cv2.INTER_LINEAR)
            elif op == "blur":
                a = cv2.GaussianBlur(a, (0, 0), random.uniform(*self.sigma))
            else:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, random.randint(*self.quality)])
                a = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return Image.fromarray(a)


class FaceFolder(Dataset):
    def __init__(self, root, max_ids=None, min_imgs=2, seed=0, augment=True, degrade_p=0.0):
        ids = sorted(d.name for d in os.scandir(root) if d.is_dir())
        if max_ids and max_ids < len(ids):                    # reproducible subset for quick runs
            ids = sorted(random.Random(seed).sample(ids, max_ids))
        self.samples, self.classes = [], []
        for name in ids:
            d = os.path.join(root, name)
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))
            if len(files) < min_imgs:
                continue
            label = len(self.classes)
            self.classes.append(name)
            self.samples += [(os.path.join(d, f), label) for f in files]
        if not self.samples:
            raise RuntimeError(f"no images found under {root}")
        ops = [T.RandomHorizontalFlip()]
        if augment and degrade_p > 0:
            ops.append(RandomDegrade(degrade_p))
        if augment:
            ops += [T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2)], p=0.5),
                    T.RandomGrayscale(p=0.05)]
        ops += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        self.tf = T.Compose(ops)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if img.size != (112, 112):
            img = img.resize((112, 112))
        return self.tf(img), label

    @property
    def n_classes(self):
        return len(self.classes)


def make_loader(ds, batch_size, workers, shuffle=True):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=shuffle,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)

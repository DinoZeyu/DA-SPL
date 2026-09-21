"""Strict read-only adapter for the retained reconstructed data, not historical folds."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SPECIAL = ('<pad>', '<unk>', '<start>', '<end>')


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_vocabulary(mapping, is_tags=False):
    if (not mapping or any(type(value) is not int for value in mapping.values()) or
            set(mapping.values()) != set(range(len(mapping))) or not all(key in mapping for key in SPECIAL)):
        raise ValueError('Vocabulary must have unique contiguous integer IDs and all special tokens')
    if is_tags and {mapping[key] for key in SPECIAL} != set(range(len(mapping) - 4, len(mapping))):
        raise ValueError('Tag specials must follow the actual multi-label classes')


class ReportDataset(Dataset):
    def __init__(self, directory, split, words, tags):
        directory = Path(directory)
        self.image_path = directory / f'{split}_IMAGES_glaucoma.hdf5'
        self.captions = read_json(directory / f'{split}_CAPTIONS_glaucoma.json')
        self.lengths = read_json(directory / f'{split}_CAPLENS_glaucoma.json')
        self.labels = read_json(directory / f'{split}_TAGSYN_glaucoma.json')
        self.records = read_json(directory / f'{split}_records.json')
        self.handle = None
        count = len(self.captions)
        if not count or any(len(values) != count for values in (self.lengths, self.labels, self.records)):
            raise ValueError(f'{split}: inconsistent sample counts')
        if len({row['id'] for row in self.records}) != count:
            raise ValueError(f'{split}: duplicate sample IDs')
        width = len(self.captions[0])
        for caption, length, labels in zip(self.captions, self.lengths, self.labels):
            if type(length) is not int or not 2 <= length <= width or len(caption) != width:
                raise ValueError(f'{split}: invalid caption length/padding')
            if any(type(token) is not int or not 0 <= token < len(words) for token in caption):
                raise ValueError(f'{split}: invalid token ID')
            if (caption[0] != words['<start>'] or caption[length - 1] != words['<end>'] or
                    caption[:length].count(words['<start>']) != 1 or caption[:length].count(words['<end>']) != 1 or
                    words['<pad>'] in caption[:length] or any(token != words['<pad>'] for token in caption[length:])):
                raise ValueError(f'{split}: invalid START/END/PAD placement')
            if len(labels) != len(tags) - 4 or any(value not in (0, 1) for value in labels):
                raise ValueError(f'{split}: invalid multi-label vector')
        with h5py.File(self.image_path, 'r') as handle:
            if (handle['images'].shape != (count, 3, 224, 224) or handle['images'].dtype != np.uint8 or
                    handle.attrs.get('captions_per_image') != 1):
                raise ValueError(f'{split}: expected one report per uint8 RGB 224x224 image')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.image_path, 'r')
        image = torch.from_numpy(self.handle['images'][index]).float() / 255
        mean = image.new_tensor([.485, .456, .406])[:, None, None]
        std = image.new_tensor([.229, .224, .225])[:, None, None]
        return {'image': (image - mean) / std, 'caption': torch.tensor(self.captions[index]),
                'length': torch.tensor(self.lengths[index]),
                'tags': torch.tensor(self.labels[index], dtype=torch.float32), 'id': self.records[index]['id']}

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def load_dataset(directory):
    directory = Path(directory)
    words = read_json(directory / 'WORDMAP_glaucoma.json')
    tags = read_json(directory / 'TAGMAP_glaucoma.json')
    validate_vocabulary(words)
    validate_vocabulary(tags, is_tags=True)
    splits = {name: ReportDataset(directory, name, words, tags) for name in ('TRAIN', 'VAL', 'TEST')}
    seen_ids, seen_pixels = set(), set()
    for name, split in splits.items():
        ids = {row['id'] for row in split.records}
        pixels = {row['pixel_sha256'] for row in split.records if row.get('pixel_sha256')}
        if ids & seen_ids or pixels & seen_pixels:
            raise ValueError(f'{name}: sample or exact pixel duplicate crosses a split boundary')
        seen_ids |= ids
        seen_pixels |= pixels
    files = sorted(directory.glob('*.json')) + sorted(directory.glob('*.hdf5'))
    hashes = {path.name: fingerprint(path) for path in files}
    return words, tags, splits, hashes

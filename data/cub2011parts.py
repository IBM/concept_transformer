# Based off of https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/cub2011.py
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from tqdm import tqdm


class CUB2011Parts_dataset(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    url = 'https://data.deepai.org/CUB200(2011).zip'
    filename = 'CUB_200_2011.tgz'
    base_folder = 'CUB_200_2011/images'
    cropped_folder = 'CUB_200_2011/cropped'

    # Grid dimensions for patches
    grid = (14, 14)
    num_patches = grid[0] * grid[1]

    # Filter to keep only attributes which occur in >=45% of all samples of a given class and which occur in >8 classes
    attributes_to_retain = [2,5,7,8,11,15,16,21,22,24,25,26,30,31,36,37,39,41,45,46,51,52,54,55,57,58,60,64,70,71,73,
                            76,81,91,92,102,105,107,111,112,117,118,120,126,127,132,133,135,146,150,152,153,154,158,159,
                            164,165,166,169,173,179,180,183,184,188,189,194,195,197,203,204,209,210,212,219,221,222,226,
                            228,236,237,239,241,244,245,249,250,254,255,260,261,263,269,275,278,284,290,293,294,295,299,
                            300,305,306,307,309,311,312]

    # Mapping non-spatial attribute id to list index position, and matching attributes to keep
    non_spatial_attributes_pos = []
    non_spatial_attributes_pos.extend(range(218, 237))
    non_spatial_attributes_pos.extend(range(249, 264))
    non_spatial_attributes_pos = list(set(non_spatial_attributes_pos).intersection(set(attributes_to_retain)))

    # Mapping spatial attribute id to list index position, and matching attributes to keep
    spatial_attributes_pos = []
    spatial_attributes_pos.extend(range(1, 218))
    spatial_attributes_pos.extend(range(237, 249))
    spatial_attributes_pos.extend(range(264, 313))
    spatial_attributes_pos = list(set(spatial_attributes_pos).intersection(set(attributes_to_retain)))

    num_non_spatial_attributes = len(non_spatial_attributes_pos)
    num_spatial_attributes = len(spatial_attributes_pos)

    def __init__(self, root='~/data/cub2011', train=True, crop=True,
                 transform=None, target_transform=None, download=False, resize_size=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        self.crop = crop

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        # Process images by cropping them
        if self.crop:
            if not os.path.exists(os.path.join(self.root, self.cropped_folder)):
                os.mkdir(os.path.join(self.root, self.cropped_folder))
                self._crop_images()
            self.sample_folder = self.cropped_folder
        else:
            self.sample_folder = self.base_folder

        # Splitting training and test data
        self._split_train_test()

    def _load_metadata(self):
        # Load images paths
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')

        # Load bounding boxes
        bbox = self._load_bounding_boxes(self.root)
        data = data.merge(bbox, on='img_id')

        # Load parts
        self.parts = self._load_parts(self.root)

        # Load class names
        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()

        self.num_classes = len(self.class_names)

        attr_list, attributes, attr_id2parts = self._load_attributes(self.root)

        # Match attribute ids and part ids
        attr_ids_part_ids = []
        for attr_id, part_ids in attr_id2parts.items():
            for part_id in part_ids:
                attr_ids_part_ids.append([attr_id, part_id])

        attr_id2parts_df = pd.DataFrame(attr_ids_part_ids, columns=('attr_id', 'part_id'))

        # Filter attributes that are present and add all possible "candidate" parts
        present_attributes = attributes.loc[attributes['is_present'] == 1][['img_id','attr_id']]
        present_attributes_candidate_parts = present_attributes.merge(attr_id2parts_df, on='attr_id')

        # Filter visible parts
        visible_parts = self.parts.loc[self.parts['visible'] == 1][['img_id','part_id','x','y']]

        # Get part-level and global attributes
        present_attributes_parts = present_attributes_candidate_parts.merge(visible_parts, on=('img_id', 'part_id'))
        present_attributes_global = present_attributes_candidate_parts[present_attributes_candidate_parts['part_id'] == 0]
        present_attributes_all = pd.concat([present_attributes_parts, present_attributes_global]).sort_values(by=['img_id','attr_id'])

        # Apply attribute filter
        present_attributes_all = present_attributes_all[present_attributes_all['attr_id'].isin(self.attributes_to_retain)]

        # Set type to int32 for id values
        self.present_attributes = present_attributes_all.astype({'img_id': 'int32', 'attr_id': 'int32', 'part_id': 'int32'})

        # Add train and test ids
        self.present_attributes = self.present_attributes.merge(train_test_split, on='img_id')

        # Add init values for transformed coordinates
        self.present_attributes['x_t'] = float("NaN")
        self.present_attributes['y_t'] = float("NaN")

        # Split train and test
        self.data = data.merge(train_test_split, on='img_id')

    def _split_train_test(self):
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
            self.present_attributes = self.present_attributes[self.present_attributes.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
            self.present_attributes = self.present_attributes[self.present_attributes.is_training_img == 0]

    @staticmethod
    def _load_bounding_boxes(root):
        bbox = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'bounding_boxes.txt'), sep=' ',
                           names=['img_id', 'x1', 'y1', 'w', 'h'])
        bbox.img_id = bbox.img_id.astype(int)
        bbox["x2"] = bbox.x1 + bbox.w
        bbox["y2"] = bbox.y1 + bbox.h
        return bbox

    @staticmethod
    def _load_parts(root):
        parts = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'parts', 'part_locs.txt'),
                            sep=' ', names=['img_id', 'part_id', 'x', 'y', 'visible'])
        parts.img_id = parts.img_id.astype(int)
        parts.part_id = parts.part_id.astype(int)
        parts.visible = parts.visible.astype(bool)
        return parts

    @staticmethod
    def _load_attributes(root):
        # Load list of attributes
        attr_list = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'attributes.txt'),
                                sep=' ', names=['attr_id', 'def'])

        # Find parts corresponding to each attribute
        has2part = {
            'bill': [2],
            'wing': [9, 13],
            'has_size': [0],   # zero means non-spatial attribute, starts with "has_size::"
            'has_shape': [0],  # zero means non-spatial attribute, starts with "has_shape::"
            'upperparts': [1, 10],
            'underparts': [3, 4, 15],
            'breast': [4],
            'back': [1],
            'tail': [14],
            'head': [5],
            'throat': [15],  # was 'throat': [17] hefore (typo?)
            'eye': [7, 11],
            'forehead': [6],
            'nape': [10],
            'belly': [3],
            'leg': [8, 12],
            'has_primary_color': [0],  # zero means non-spatial attribute, starts with "has_primary_color::"
            'crown': [5],
        }
        attr_id2parts = defaultdict(list)
        for has, part in has2part.items():
            # need to check only attribute categories (left of "::" in the source), not attribute category values
            # see e.g. leg in "225 has_shape::long-legged-like"
            attr_id = attr_list[attr_list['def'].str.split('::').str[0].str.contains(has)].attr_id
            for k in attr_id:
                attr_id2parts[k] += part

        # Load attributes of each image
        attributes = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'),
                                 sep=' ', names=['img_id', 'attr_id', 'is_present', 'certainty', 'time'])
        attributes.img_id = attributes.img_id.astype(int)
        attributes.attr_id = attributes.attr_id.astype(int)
        attributes.is_present = attributes.is_present.astype(int)

        return attr_list, attributes, attr_id2parts

    def _get_explanations(self, idx, current_attrs_parts, img_shape):
        # Global explanations
        expl = torch.FloatTensor(self.num_non_spatial_attributes).fill_(float('nan'))

        # Spatial (patch-level) explanantions
        spatial_expl = torch.FloatTensor(self.num_patches, self.num_spatial_attributes).fill_(float('nan'))

        # get all parts with coordinates for current image
        for index, row in current_attrs_parts.iterrows():
            # part_id equal zero means non-spatial, i.e. global attribute
            if row.part_id == 0:
                expl_index = self.non_spatial_attributes_pos.index(int(row.attr_id))
                expl[expl_index] = 1
                # If there is at least one non-spatial (global) attribute, nans become 0s.
                expl[expl != expl] = 0
                continue

            img_width = img_shape[1]
            img_height = img_shape[2]

            # The potentially transformed x,y coordinates must lie within the image
            # (left and top edges are inside, right and bottom edges are outside)
            if row.x_t < 0 or row.x_t >= img_width or \
                    row.y_t < 0 or row.y_t >= img_height:
                continue

            patch_number = self._get_patch_number(row.x_t, row.y_t, img_width, img_height, self.grid)

            spatial_expl_index = self.spatial_attributes_pos.index(int(row.attr_id))
            spatial_expl[patch_number][spatial_expl_index] = 1
            # If there is at least one (spatial) attribute on the current patch, nans become 0s.
            spatial_expl[patch_number][spatial_expl[patch_number] != spatial_expl[patch_number]] = 0

        return expl, spatial_expl

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        # See https://github.com/pytorch/vision/issues/4156
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        print('Downloading dataset')
        download_and_extract_archive(self.url, self.root)
        root = os.path.expanduser(self.root)
        extract_archive(os.path.join(root, self.filename), root, True)

        # Move `attributes.txt` to basefolder instead of root
        if os.path.exists(os.path.join(self.root, 'attributes.txt')):
            os.rename(os.path.join(self.root, 'attributes.txt'),
                      os.path.join(self.root, 'CUB_200_2011', 'attributes.txt'))

    def _crop_images(self):
        """Process images by cropping them and saving them in `cropped_folder`
        """
        for img_id, sample in tqdm(self.data.iterrows(), total=len(self.data),
                                   desc="Cropping and saving images..."):
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            img = self.loader(path)

            proc_img = img.crop(sample[['x1', 'y1', 'x2', 'y2']])
            proc_filename = os.path.join(self.root, self.cropped_folder, sample.filepath)
            if not os.path.exists(os.path.dirname(proc_filename)):
                os.mkdir(os.path.dirname(proc_filename))
            proc_img.save(proc_filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (PIL image, target) where target is index of the target class.
        """
        sample_metadata = self.data.iloc[idx]
        image_id = self.data.iloc[idx]['img_id']
        path = os.path.join(self.root, self.sample_folder, sample_metadata.filepath)
        target = sample_metadata.target - 1  # Targets start at 1 by default, so shift to 0

        # get all parts with coordinates for current image
        bbox_x = sample_metadata.x1
        bbox_y = sample_metadata.y1

        current_attrs_parts = self.present_attributes[self.present_attributes['img_id'] == image_id]
        part_coordinates = []
        for index, row in current_attrs_parts.iterrows():
            # row.x, row.y are part coordinates
            if row.part_id == 0:
                current_attrs_parts.at[index, 'x'] = 0.0
                current_attrs_parts.at[index, 'y'] = 0.0
                part_coordinates.append((0.0, 0.0))
            else:
                part_coordinates.append((row.x - bbox_x, row.y - bbox_y))

        img = self.loader(path)

        if self.transform is not None:
            image_np = np.array(img)
            transformed = self.transform(image=image_np, keypoints=part_coordinates)
            img = transformed['image']
            kps = transformed['keypoints']

            kps_idx = 0
            for i, row in current_attrs_parts.iterrows():
                current_attrs_parts.at[i, 'x_t'] = int(kps[kps_idx][0])
                current_attrs_parts.at[i, 'y_t'] = int(kps[kps_idx][1])
                kps_idx += 1

        if self.target_transform is not None:
            target = self.target_transform(target)

        expl, spatial_expl = self._get_explanations(idx, current_attrs_parts, img.shape)

        return (img, expl, spatial_expl, target)

    def _get_patch_number(self, x, y, width, height, grid):
        """Patch numbers are assigned by row, patch 0 is top left; patch 1
           is 2nd column, 1st row; etc.
           Precorditions:
           - width/height of grid is smaller than width/height of image
        """
        patch_x = int(x * grid[0] / width)
        patch_y = int(y * grid[1] / height)

        return patch_x + (patch_y * grid[1])

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from IPython.display import Markdown, display


def printmd(string):
    display(Markdown(string))

def batch_predict_results(dat_list):
    dat = defaultdict(list)
    for d in dat_list:
        for k, v in d.items():
            if v is not None:
                dat[k] += v

    for k in dat.keys():
        if isinstance(dat[k][0], torch.Tensor):
            dat[k] = torch.cat([d.unsqueeze(0) for d in dat[k]])
            dat[k] = dat[k].squeeze().cpu()
    return dat


def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_explanation(raster, ax):
    length = raster.shape[-1]
    ax.imshow(raster)
    # grid
    ax.set_xticks(np.arange(.5, length - 0.5, 1), minor=True)
    ax.set_yticks(np.arange(.5, 1, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    # Remove y-axis
    ax.get_yaxis().set_visible(False)
    ax.set_xticks(range(length))
    remove_spines(ax)


def unnorm_cub(img):
    sd = np.array([0.229, 0.224, 0.225])
    mu = np.array([0.485, 0.456, 0.406])
    img = img.transpose(0, 2).transpose(0, 1)
    return img * sd + mu

def plot_cub_gt(sample):
    """
        plot_cub(data_module.cub_test[2])
    """
    img, expl, spatial_expl, label = sample

    im = Image.fromarray(np.uint8(unnorm_cub(img)*255)).convert("RGBA")

    n_patch = int(np.sqrt(spatial_expl.shape[0]))
    patch_idx = ~torch.isnan(spatial_expl[:,0])
    patches = np.zeros(n_patch**2) + 0.3
    patches[patch_idx] = 1.0
    patches = patches.reshape(n_patch, n_patch)

    im_p = Image.fromarray(np.uint8(patches * 255)).convert("L")
    im_p = im_p.resize(im.size, Image.ANTIALIAS)

    im.putalpha(im_p)

    plt.imshow(im)
    plt.axis('off')
    plt.show()

def plot_cub_expl(results, ind, data_module):
    idx = results['idx'][ind].item()

    img = data_module.cub_test[idx][0]
    im = Image.fromarray(np.uint8(unnorm_cub(img)*255)).convert("RGBA")

    # Prediction
    pred = results['preds'][ind].item()
    prediction = data_module.cub_test.class_names[pred].split('/')[0][4:]

    # Spatial attention
    attn = results['spatial_concept_attn'][ind]
    n_patch = int(np.sqrt(attn.shape[0]))

    # Get most active patches
    patch_idx = attn.max(axis=1)[0] > 0.6

    patches = np.zeros(n_patch**2) + 0.4
    patches[patch_idx] = 1.0
    patches = patches.reshape(n_patch, n_patch)

    # Get corresponding most active attributes
    attr_idx = attn[patch_idx,:].max(axis=0)[0] > 0.3
    attr_ind = np.nonzero(attr_idx)

    attr_list = load_attributes()
    attributes = attr_list[np.array(data_module.cub_test.spatial_attributes_pos)[attr_ind] - 1]

    # Nonspatial explanation
    expl = results['concept_attn'][ind]
    expl_idx = expl > 0.2
    nonspatial_attributes = attr_list[np.array(data_module.cub_test.non_spatial_attributes_pos)[expl_idx] - 1]

    # Plot
    im_p = Image.fromarray(np.uint8(patches * 255)).convert("L")
    im_p = im_p.resize(im.size, Image.ANTIALIAS)

    im.putalpha(im_p)

    plt.imshow(im)
    plt.axis('off')
    plt.show()

    correct = results['correct'][ind].item()
    correct_wrong = ['*wrong*', '*correct*'][correct]
    if not correct:
        gt = data_module.cub_test[idx][3]
        gt = data_module.cub_test.class_names[gt].split('/')[0][4:]
        correct_wrong += f', gt is {gt}'

    printmd(f'**Prediction**: {prediction} ({correct_wrong})')

    print(' Spatial explanations:')
    if isinstance(attributes, str):
        attributes = [[attributes]]
    for t in attributes:
        print(f'   - {t[0]}')

    print(' Global explanations:')
    for t in nonspatial_attributes:
        print(f'   - {t}')

def load_attributes(root='/data/Datasets/'):
    # Load list of attributes
    attr_list = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'attributes.txt'),
                            sep=' ', names=['attr_id', 'def'])
    attr_list = np.array(attr_list['def'].to_list())
    return attr_list

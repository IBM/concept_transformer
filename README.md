# Concept Transformer

Pytorch code for ConceptTransformer architecture presented in paper
> Mattia Rigotti, Christoph Miksovic, Ioana Giurgiu, Thomas Gschwind, Paolo Scotton, "Attention-based Interpretability with Concept Transformers",
in International Conference on Learning Representations (ICLR), 2022 [[pdf]](https://openreview.net/pdf?id=kAa9eDS0RdO)

## Requirements

* torch==1.10.0
* torchvision==0.11.1
* pytorch-lightning==1.4.8
* lightning-bolts==0.3.4
* torchmetrics==0.5
* scipy==1.7.1
* numpy==1.20.3
* pandas==1.3.3
* albumentations==1.0.3
* timm==0.4.12
* setuptools==59.5.0

These can be installed using `pip` by running:

```bash
pip install -r requirements.txt
```


## Usage

### MNIST even/odd

Run the code on the *MNIST even/odd* dataset with

```bash
python ctc_mnist.py
```
Get help on available arguments with
```bash
python ctc_mnist.py --help
```

```bash
usage: ctc_mnist.py [-h] [--weight-decay WEIGHT_DECAY] [--attention_sparsity ATTENTION_SPARSITY] [--disable_lr_scheduler]
                    [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--max_epochs MAX_EPOCHS] [--warmup WARMUP]
                    [--expl_lambda EXPL_LAMBDA] [--n_train_samples N_TRAIN_SAMPLES]

Training with explanations on MNIST Even/Odd

optional arguments:
  -h, --help            show this help message and exit
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 1e-4)
  --attention_sparsity ATTENTION_SPARSITY
                        sparsity penalty on attention
  --disable_lr_scheduler
                        disable cosine lr schedule
  --batch_size BATCH_SIZE
  --learning_rate LEARNING_RATE
  --max_epochs MAX_EPOCHS
  --warmup WARMUP
  --expl_lambda EXPL_LAMBDA
  --n_train_samples N_TRAIN_SAMPLES
```


 <img src="/figs/binary_mnist_correct.png" width="300">
 <img src="/figs/binary_mnist_error.png" width="300">

### Concept Transformer on VIT for CUB-200-2011 dataset

Run the code on the *CUB-200-2011* dataset with

```bash
python cvit_cub.py
```
(this requires a GPU).

Get help on available arguments with
```bash
python cvit_cub.py --help
```

```bash
usage: cvit_cub.py [-h] [--weight-decay WEIGHT_DECAY] [--learning_rate LEARNING_RATE]
                   [--batch_size BATCH_SIZE] [--debug] [--data_dir DATA_DIR]
                   [--weight_decay WEIGHT_DECAY]
                   [--attention_sparsity ATTENTION_SPARSITY]
                   [--max_epochs MAX_EPOCHS] [--warmup N]
                   [--finetune_unfreeze_epoch N] [--disable_lr_scheduler]
                   [--baseline] [--expl_lambda EXPL_LAMBDA]
                   [--num_workers NUM_WORKERS]

CUB dataset with explanations

optional arguments:
  -h, --help            show this help message and exit
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 1e-4)
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --debug               Set debug mode in Lightning module
  --data_dir DATA_DIR   dataset root directory
  --weight_decay WEIGHT_DECAY
                        weight decay (default: 1e-5)
  --attention_sparsity ATTENTION_SPARSITY
                        sparsity penalty on attention
  --max_epochs MAX_EPOCHS
  --warmup N            number of warmup epochs
  --finetune_unfreeze_epoch N
                        Epoch until when to finetune classifier head before
                        unfreeezing feature extractor
  --disable_lr_scheduler
                        disable cosine lr schedule
  --baseline            run baseline without concepts
  --expl_lambda EXPL_LAMBDA
                        weight of explanation loss
  --num_workers NUM_WORKERS
                        number of workers
```

 <img src="/figs/cub_examples.png" width="800">


## Citation
> Mattia Rigotti, Christoph Miksovic, Ioana Giurgiu, Thomas Gschwind, Paolo Scotton, "Attention-based Interpretability with Concept Transformers",
in International Conference on Learning Representations (ICLR), 2022 [[OpenReview](https://openreview.net/pdf?id=kAa9eDS0RdO)]




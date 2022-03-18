import os
from argparse import ArgumentParser

import data
import pytorch_lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torchmetrics.functional.classification.accuracy import accuracy

import ctc
from ctc import concepts_cost, concepts_sparsity_cost, spatial_concepts_cost


class CTCModel(pl.LightningModule):
    def __init__(
        self,
        ctc_model,
        expl_lambda,
        learning_rate,
        weight_decay,
        warmup,
        disable_lr_scheduler,
        max_epochs,
        attention_sparsity,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters()

        # Instantiate CTC model
        self.model = getattr(ctc, ctc_model)(**kwargs)

    @staticmethod
    def get_model_args(parser=None):
        """Args that have to do with the model training"""
        parser = ArgumentParser(parents=[parser] if parser is not None else [], add_help=False)
        parser.add_argument("--learning_rate", default=0.0004, type=float)
        parser.add_argument(
            "--weight-decay", default=1e-3, type=float, help="weight decay (default: 1e-4)"
        )
        parser.add_argument(
            "--attention_sparsity", default=0.0, type=float, help="sparsity penalty on attention"
        )
        parser.add_argument("--max_epochs", default=150, type=int)
        parser.add_argument(
            "--warmup", default=50, type=int, metavar="N", help="number of warmup epochs"
        )
        parser.add_argument(
            "--disable_lr_scheduler", action="store_true", help="disable cosine lr schedule"
        )
        parser.add_argument(
            "--expl_lambda", default=5.0, type=float, help="weight of explanation loss"
        )
        return parser

    def forward(self, x):
        """Returns classification logits and concepts

        Outputs:
            logits: torch.Tensor (batch, n_classes)
            concept_attn: torch.Tensor (batch, n_concepts)
            spatial_concept_attn: torch.Tensor (batch, n_tokens, n_concepts)
        """
        outputs, unsup_concept_attn, concept_attn, spatial_concept_attn = self.model(x)
        logits = F.log_softmax(outputs, dim=-1)
        return logits, unsup_concept_attn, concept_attn, spatial_concept_attn

    def training_step(self, batch, batch_idx):
        logits, preds, ce_loss, acc, expl_loss, l1_loss, _, _, _ = self.shared_step(batch)
        loss = ce_loss
        if self.hparams.expl_lambda > 0:
            loss = loss + self.hparams.expl_lambda * expl_loss
        if self.hparams.attention_sparsity > 0:
            loss = loss + self.hparams.attention_sparsity * l1_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_ce_loss": ce_loss,
                "train_acc": acc,
                "train_expl_loss": expl_loss,
                "train_l1_loss": l1_loss,
            },
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits, preds, ce_loss, acc, expl_loss, l1_loss, _, _, _ = self.shared_step(batch)
        self.log_dict(
            {
                "val_ce_loss": ce_loss,
                "val_acc": acc,
                "val_expl_loss": expl_loss,
                "val_l1_loss": l1_loss,
            },
            on_step=False,
            on_epoch=True,
        )
        return acc

    def test_step(self, batch, batch_idx):
        logits, preds, ce_loss, acc, expl_loss, l1_loss, _, _, _ = self.shared_step(batch)
        self.log_dict(
            {
                "test_ce_loss": ce_loss,
                "test_acc": acc,
                "test_expl_loss": expl_loss,
                "test_l1_loss": l1_loss,
            }
        )
        return acc

    def predict_step(self, batch, batch_idx):
        x, expl, spatial_expl, y = batch
        (
            logits,
            preds,
            ce_loss,
            acc,
            expl_loss,
            l1_loss,
            unsup_concept_attn,
            concept_attn,
            spatial_concept_attn,
        ) = self.shared_step(batch)

        correct = preds == y
        idx = batch_idx * len(x) + torch.arange(len(x))
        return {
            "idx": idx,
            "preds": preds,
            "y": y,
            "correct": correct,
            "unsup_concept_attn": unsup_concept_attn,
            "concept_attn": concept_attn,
            "expl": expl,
            "spatial_concept_attn": spatial_concept_attn,
            "spatial_expl": spatial_expl,
        }

    def shared_step(self, batch):
        x, expl, spatial_expl, y = batch
        logits, unsup_concept_attn, concept_attn, spatial_concept_attn = self(x)
        preds = torch.argmax(logits, dim=1)

        ce_loss = F.nll_loss(logits, y)
        acc = accuracy(preds, y)

        expl_loss = concepts_cost(concept_attn, expl) + spatial_concepts_cost(
            spatial_concept_attn, spatial_expl
        )
        l1_loss = concepts_sparsity_cost(concept_attn, spatial_concept_attn)

        return (
            logits,
            preds,
            ce_loss,
            acc,
            expl_loss,
            l1_loss,
            unsup_concept_attn,
            concept_attn,
            spatial_concept_attn,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.disable_lr_scheduler:
            return [optimizer]
        else:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.hparams.warmup, max_epochs=self.hparams.max_epochs
            )
        return [optimizer], [scheduler]


def run_exp(args):
    print("\n* RUN EXPERIMENT {}\n".format(args))

    model = CTCModel(**vars(args))
    data_module = getattr(data, args.data_name)(**vars(args))

    project_name = getattr(args, "project_name", args.ctc_model)
    run_name = getattr(args, "run_name", f"{args.data_name}_expl{args.expl_lambda}")

    logger = TensorBoardLogger(".", name="logs")

    # Callbacks
    best_checkpoint = os.path.join(".", project_name, run_name, "binary_mnist_best_ckpt")
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.dirname(best_checkpoint),
            monitor="val_ce_loss",
            filename=os.path.basename(best_checkpoint),
        )
    ]

    if getattr(args, "early_stopping_patience", 0) > 0:
        callbacks += [
            EarlyStopping("val_ce_loss", patience=args.early_stopping_patience, verbose=True)
        ]

    # Train only head
    if getattr(args, "finetune_unfreeze_epoch", 0) > 0:
        model.model.feature_extractor.requires_grad_(False)
        finetune_trainer = get_trainer(
            args.finetune_unfreeze_epoch, logger, callbacks, debug=getattr(args, "debug", False)
        )
        finetune_trainer.fit(model, data_module)
        model.model.feature_extractor.requires_grad_(True)

    # Train the whole model
    trainer = get_trainer(
        args.max_epochs, logger, callbacks, amp=True, debug=getattr(args, "debug", False)
    )
    trainer.fit(model, data_module)

    return model, trainer


def get_trainer(max_epochs, logger, callbacks, amp=False, debug=False):
    if debug:
        trainer = pl.Trainer(
            fast_dev_run=True,
            weights_summary="full",
            log_every_n_steps=1,
            logger=logger,
            max_epochs=max_epochs,
            callbacks=callbacks,
            progress_bar_refresh_rate=10,
        )
    else:
        if torch.cuda.device_count():
            if amp:
                kwargs = {"amp_backend": "apex", "amp_level": "O2", "precision": 16}
            else:
                kwargs = {}

            trainer = pl.Trainer(
                logger=logger,
                gpus=-1,
                auto_select_gpus=True,
                max_epochs=max_epochs,
                callbacks=callbacks,
                progress_bar_refresh_rate=10,
                gradient_clip_val=1.0,
                **kwargs,
            )
        else:
            trainer = pl.Trainer(
                logger=logger,
                max_epochs=max_epochs,
                callbacks=callbacks,
                progress_bar_refresh_rate=10,
            )
    return trainer


def load_exp(run_path):
    # Find checkpoint file
    if run_path[-5:] != ".ckpt":
        ckpts = []
        for path, subdirs, files in os.walk(run_path):
            for name in files:
                if name[-5:] == ".ckpt":
                    ckpts.append(os.path.join(path, name))
        if len(ckpts) > 1:
            print(f"Found more than 1 checkpoint. Loading {ckpts[0]}")
        if len(ckpts) == 0:
            raise ValueError(f"Could not find any checkpoints in {run_path}")
        run_path = ckpts[0]

    # Load model
    model = CTCModel.load_from_checkpoint(run_path)

    # Load data_module
    data_module = getattr(data, model.hparams.data_name)(**model.hparams)
    data_module.prepare_data()
    data_module.setup()

    return model, data_module

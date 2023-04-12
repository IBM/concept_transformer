from argparse import Namespace
import numpy as np
import torch
from epyc import Experiment, JSONLabNotebook, Lab

from ctc import CTCModel, run_exp


class BinaryMNISTS(Experiment):
    def do(self, params):

        args = Namespace()
        args.__dict__ = params

        # Set random seeds
        torch.manual_seed(args.run)
        np.random.seed(args.run)

        args = self.add_dependent_args(args)

        # Start experiment
        model, trainer = run_exp(args)
        test_results = trainer.test(ckpt_path='best')
        trainer.logger.experiment.finish()
        del trainer, model
        return test_results[0]

    @staticmethod
    def add_dependent_args(args):
        args.run_name = f"expl{args.expl_lambda}_N{args.n_train_samples}"
        return args


PROJECT_NAME = "binary_mnist_scaling_ES"
lab = Lab(notebook=JSONLabNotebook(
    f"experiments/{PROJECT_NAME}.json",
    create=True,
    description="Effect of explanations on sample complexity for BinaryMNIST",
))

lab["project_name"] = PROJECT_NAME
lab["ctc_model"] = "mnist_ctc"
lab["data_name"] = "ExplanationMNIST"

# Add default model params
default_params = vars(CTCModel.get_model_args().parse_args())
for k, v in default_params.items():
    lab.addParameter(k, v)

# Learning params
lab["learning_rate"] = 0.0004
lab["max_epochs"] = 150
lab["warmup"] = 20
lab["batch_size"] = 32
lab["early_stopping_patience"] = 15

# Varying parameters
lab["expl_lambda"] = [0.0, 2.0]
lab["n_train_samples"] = (list(range(100, 1000, 100)) +
                          list(range(1000, 11000, 1000)))
lab["run"] = [4, 5]


# Run experiments
lab.runExperiment(BinaryMNISTS())

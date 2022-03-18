from argparse import ArgumentParser
from ctc import CTCModel, run_exp

CTC_MODEL = 'mnist_ctc'
DATA_NAME = 'ExplanationMNIST'


def get_parser(parser):
    parser = ArgumentParser(description='Training with explanations on MNIST Even/Odd',
                            parents=[parser], conflict_handler='resolve')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--expl_lambda', default=5.0, type=float)
    parser.add_argument('--n_train_samples', default=55000, type=int,
                        help='number of MNIST samples to be used for training')
    return parser


parser = CTCModel.get_model_args()
parser = get_parser(parser)
args = parser.parse_args()

args.ctc_model = CTC_MODEL
args.data_name = DATA_NAME

model, trainer = run_exp(args)
test_results = trainer.test()

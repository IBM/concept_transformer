from argparse import ArgumentParser
from ctc import CTCModel, run_exp

CTC_MODEL = 'cub_cvit'
DATA_NAME = 'CUB2011Parts'


def get_parser(parser):
    parser = ArgumentParser(description='CUB dataset with explanations',
                            parents=[parser], conflict_handler='resolve')
    parser.add_argument('--learning_rate', default=0.00005, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--debug', action='store_true',
                        help='Set debug mode in Lightning module')
    parser.add_argument('--data_dir', default='~/data/cub2011/', type=str,
                        help='dataset root directory')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--attention_sparsity', default=0.5, type=float,
                        help='sparsity penalty on attention')
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--warmup', default=10, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('--finetune_unfreeze_epoch', default=0, type=int, metavar='N',
                        help='Epoch until when to finetune classifier head before unfreeezing feature extractor')
    parser.add_argument('--disable_lr_scheduler', action='store_true',
                        help='disable cosine lr schedule')
    parser.add_argument('--baseline', action='store_true',
                        help='run baseline without concepts')
    parser.add_argument('--expl_lambda', default=1.0, type=float,
                        help='weight of explanation loss')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of workers')
    return parser


parser = CTCModel.get_model_args()
parser = get_parser(parser)
args = parser.parse_args()

args.ctc_model = CTC_MODEL
args.data_name = DATA_NAME

model, trainer = run_exp(args)
test_results = trainer.test()

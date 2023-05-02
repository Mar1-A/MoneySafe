"""
Author: Marwan Alalloush
"""
import argparse

parser = argparse.ArgumentParser(description='Speech Recognition Training')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--global_step', type=int, default=0, help='Global step')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.001)')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout')
parser.add_argument('--recurrent_dropout', type=float, default=0.5, help='recurrent_dropout')
parser.add_argument('--num_features', type=int, default=39, help='number of audio features')
parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden layers')
parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
parser.add_argument('--num_classes', type=int, default=29, help='number of classes to pridict')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--beam_width', type=int, default=10, help='beam width')
parser.add_argument('--steps_per_checkpoint', type=int, default=160, help='steps per checkpoint')
parser.add_argument('--checkpoint_num', type=int, default=0, help='checkpoint Number')
parser.add_argument('--checkpoints_dirpath', type=str, default='C:/Users/Shadow/Desktop/Money_safe/model/checkpoints', help='path for check points')
parser.add_argument('--hparams_path', type=str, default='model/hparams', help='hparams directory path')
parser.add_argument('--train_path', type=str, default='Data/librispeech_processed/train-clean-100', help='training data directory path')
parser.add_argument('--test_path', type=str, default='Data/librispeech_processed/test-clean', help='testing data directory path')
parser.add_argument('--log_dir', type=str, default='model/logs', help='testing data directory path')
parser.add_argument('--out_dir', type=str, default='DBN_model', help='testing data directory path')
parser.add_argument('--load_from_checkpoint', type=bool, default=False, help='load from checkpoint')
parser.add_argument('--input_max_len', type=int, default=0, help='Input signal max length')
args, unknown = parser.parse_known_args()
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--steps_per_checkpoint", type=int, default=160)
parser.add_argument("--checkpoints_dirpath", type=str, default="model/checkpoints")
parser.add_argument("--learning_rate", type=float, default=0.025)
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--max_data", type=int, default=0)
parser.add_argument("--load_from_checkpoint", type=bool, default=False)
parser.add_argument("--checkpoint_num", type=int, default=88460)

# Data config
parser.add_argument(
    "--librispeech_path", type=str, default="data/librispeech_processed"
)

# # CNN config
# parser.add_argument("--cnn_layers", type=int, default=2)

# # RNN config
# parser.add_argument("--rnn_layers", type=int, default=3)
# parser.add_argument("--rnn_size", type=int, default=128)

# # Decoder config
# parser.add_argument("--beam_width", type=int, default=32)

args = parser.parse_args()

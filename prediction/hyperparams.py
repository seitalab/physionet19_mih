import argparse

parser = argparse.ArgumentParser()

# Common
parser.add_argument('--imp', type=int, default=0)
parser.add_argument('--scaler', type=int, default=1)
parser.add_argument('--preprocess', type=int, default=4)
parser.add_argument('--cw', type=float, default=1.)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default="GRUv2")
parser.add_argument('--base-list', type=str, default=None)

# NN
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--optim', type=str, default="sgd")
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--save-every', type=int, default=5)
parser.add_argument('--loss-scale', type=float, default=10.)

parser.add_argument('--ufunc', type=str, default="v7")
parser.add_argument('--split-mode', type=str, default="v1")

parser.add_argument('--i_dim', type=int, default=40)
parser.add_argument('--h_dim', type=int, default=32)

parser.add_argument('--clip-value', type=float, default=1.)

args = parser.parse_args()
print(args)

if __name__ == "__main__":
    print("-"*80)
    print(args)

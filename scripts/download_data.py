"""Download only the selected dataset; torchvision verifies its checksums."""
import argparse
from pathlib import Path


def main():
    from torchvision import datasets
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",choices=["cifar10","fashion_mnist","mnist"])
    parser.add_argument("--data-root",default="data")
    args=parser.parse_args()
    cls={"cifar10":datasets.CIFAR10,"fashion_mnist":datasets.FashionMNIST,"mnist":datasets.MNIST}[args.dataset]
    if args.dataset=="fashion_mnist":
        cls.mirrors=["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    for train in (True,False):
        dataset=cls(Path(args.data_root),train=train,download=True)
        print(f"{args.dataset}: {'train' if train else 'test'}={len(dataset)}")


if __name__=="__main__":main()

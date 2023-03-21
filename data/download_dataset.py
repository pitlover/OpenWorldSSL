from torchvision.datasets.cifar import CIFAR10, CIFAR100

print("Download Cifar10")
CIFAR10(root="./", train=True, download=True)
CIFAR10(root="./", train=False, download=True)

print("Download Cifar10")
CIFAR100(root="./", train=True, download=True)
CIFAR100(root="./", train=False, download=True)

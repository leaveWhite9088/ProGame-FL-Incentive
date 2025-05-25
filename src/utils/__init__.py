import os

util_type = os.getenv("UTIL_TYPE", "MNIST")  # 默认用A

if util_type == "MNIST":
    from .UtilMNIST import UtilMNIST as UtilT
elif util_type == "CIFAR10":
    from .UtilCIFAR10 import UtilCIFAR10 as UtilT
elif util_type == "CIFAR100":
    from .UtilCIFAR100 import UtilCIFAR100 as UtilT
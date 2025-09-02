# ProGame-FL: Proactive Game-based Incentive Mechanism for Federated Learning

This repository contains the implementation of ProGame-FL (Proactive Game-based Incentive Mechanism for Federated Learning), a novel incentive mechanism designed for federated learning environments.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Experiments](#experiments)
  - [Incentive Verification Experiments](#incentive-verification-experiments)
  - [Selection Verification Experiments](#selection-verification-experiments)
  - [Baseline Comparison Experiments](#baseline-comparison-experiments)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 📖 Overview

ProGame-FL is an innovative incentive mechanism that addresses the challenges in federated learning by:
- Encouraging data owners to contribute high-quality data
- Optimizing resource allocation through game-theoretic approaches
- Implementing dynamic feedback learning for continuous improvement
- Providing fair payment distribution based on contribution

## 🗂️ Project Structure

```
ProGame-FL-Incentive/
├── src/
│   ├── algorithms/          # Game theory algorithms (Stackelberg, Gale-Shapley, Cournot)
│   ├── datasets/            # Dataset handling for MNIST, CIFAR-10, and CIFAR-100
│   ├── experiments/         # Experimental implementations
│   ├── models/              # CNN models for different datasets
│   ├── roles/               # Federated learning roles (DataOwner, ModelOwner, ComputingCenter)
│   ├── utils/               # Utility functions for each dataset
│   └── global_variable.py   # Global configuration variables
├── data/                    # Dataset storage and model checkpoints
└── README.md               # This file
```

## 📊 Dataset Setup

### Dataset Structure

Place the downloaded datasets in the following structure:

```
data/
└── dataset/
    ├── CIFAR10/
    │   ├── batches.meta
    │   ├── data_batch_1
    │   ├── data_batch_2
    │   ├── data_batch_3
    │   ├── data_batch_4
    │   ├── data_batch_5
    │   └── test_batch
    ├── CIFAR100/
    │   ├── meta
    │   ├── test
    │   └── train
    └── MNIST/
        ├── t10k-images.idx3-ubyte
        ├── t10k-labels.idx1-ubyte
        ├── train-images.idx3-ubyte
        └── train-labels.idx1-ubyte
```

### Dataset Downloads

#### MNIST

- [Training images](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz)
- [Training labels](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz)
- [Testing images](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz)
- [Testing labels](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

#### CIFAR10

- [CIFAR-10 Python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

#### CIFAR100

- [CIFAR-100 Python](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

## ⚙️ Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- NumPy
- Scikit-learn

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ProGame-FL-Incentive

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Experiments

### Incentive Verification Experiments

Each incentive verification experiment takes approximately 30 minutes to run.

#### Fixed Strategy
```shell
# MNIST with fixed strategy - utility
python -m src.experiments.verify_incentives.fixed.verify_incentives-fixed-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST

# MNIST with fixed strategy - utility (test)
python -m src.experiments.verify_incentives.fixed.verify_incentives-fixed-utility-MNIST-test --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST
```

#### PGI-RDFL (Our Method)
```shell
# MNIST with PGI-RDFL - utility
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST

# MNIST with PGI-RDFL - accuracy
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_incentives-accuracy --util MNIST

# MNIST with PGI-RDFL - utility (test)
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-utility-MNIST-test --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST
```

#### Random Strategy
```shell
# MNIST with random strategy - utility
python -m src.experiments.verify_incentives.random.verify_incentives-random-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST
```

### Selection Verification Experiments

Each selection verification experiment takes approximately 30 minutes to run.

#### All Selection
```shell
# MNIST with all_selection - utility
python -m src.experiments.verify_selection.all_selection.verify_selection-as-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST

# MNIST with all_selection - accuracy
python -m src.experiments.verify_selection.all_selection.verify_selection-as-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy --util MNIST
```

#### Non Two-Way Selection (Use results to replace random results)
```shell
# MNIST with non_two_way_selection - utility
python -m src.experiments.verify_selection.non_two_way_selection.verify_selection-ntws-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST

# MNIST with non_two_way_selection - accuracy
python -m src.experiments.verify_selection.non_two_way_selection.verify_selection-ntws-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy --util MNIST
```

#### Random Selection
```shell
# MNIST with random_selection - utility
python -m src.experiments.verify_selection.random_selection.verify_selection-rs-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST

# MNIST with random_selection - accuracy
python -m src.experiments.verify_selection.random_selection.verify_selection-rs-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy --util MNIST
```

#### Two-Way Selection
```shell
# MNIST with two_way_selection - utility
python -m src.experiments.verify_selection.two_way_selection.verify_selection-tws-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST

# MNIST with two_way_selection - accuracy
python -m src.experiments.verify_selection.two_way_selection.verify_selection-tws-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy --util MNIST
```

### Baseline Comparison Experiments

#### MNIST Dataset
```shell
# Baseline utility
python -m src.experiments.baseline_comparison.Utility.baseline_utility-MNIST --adjustment_literation -1 --parent_path comparison_experiments --util MNIST
# PGI-RDFL utility
python -m src.experiments.baseline_comparison.Utility.pgi_rdfl-utility-MNIST --adjustment_literation -1 --parent_path comparison_experiments --util MNIST
# Fixed utility
python -m src.experiments.baseline_comparison.Utility.fixed-utility-MNIST --adjustment_literation -1 --parent_path comparison_experiments --util MNIST
# Random utility
python -m src.experiments.baseline_comparison.Utility.random-utility-MNIST --adjustment_literation -1 --parent_path comparison_experiments --util MNIST

# Baseline accuracy
python -m src.experiments.baseline_comparison.Accuracy.baseline_accuracy-MNIST --adjustment_literation 100 --parent_path comparison_experiments --util MNIST
# PGI-RDFL accuracy
python -m src.experiments.baseline_comparison.Accuracy.pgi_rdfl-accuracy-MNIST --adjustment_literation 100 --parent_path comparison_experiments --util MNIST
# Fiexed accuracy
python -m src.experiments.baseline_comparison.Accuracy.fixed-accuracy-MNIST --adjustment_literation 100 --parent_path comparison_experiments --util MNIST
# Random accuracy
python -m src.experiments.baseline_comparison.Accuracy.random-accuracy-MNIST --adjustment_literation 100 --parent_path comparison_experiments --util MNIST
```

#### CIFAR-10 Dataset
```shell
# Baseline utility
python -m src.experiments.baseline_comparison.Utility.baseline_utility-CIFAR10 --adjustment_literation -1 --parent_path comparison_experiments --util CIFAR10
# PGI-RDFL utility
python -m src.experiments.baseline_comparison.Utility.pgi_rdfl-utility-CIFAR10 --adjustment_literation -1 --parent_path comparison_experiments --util CIFAR10
# Fixed utility
python -m src.experiments.baseline_comparison.Utility.fixed-utility-CIFAR10 --adjustment_literation -1 --parent_path comparison_experiments --util CIFAR10
# Random utility
python -m src.experiments.baseline_comparison.Utility.random-utility-CIFAR10 --adjustment_literation -1 --parent_path comparison_experiments --util CIFAR10

# Baseline accuracy
python -m src.experiments.baseline_comparison.Accuracy.baseline_accuracy-CIFAR10 --adjustment_literation 100 --parent_path comparison_experiments --util CIFAR10
# PGI-RDFL accuracy
python -m src.experiments.baseline_comparison.Accuracy.pgi_rdfl-accuracy-CIFAR10 --adjustment_literation 100 --parent_path comparison_experiments --util CIFAR10
# Fiexed accuracy
python -m src.experiments.baseline_comparison.Accuracy.fixed-accuracy-CIFAR10 --adjustment_literation 100 --parent_path comparison_experiments --util CIFAR10
# Random accuracy
python -m src.experiments.baseline_comparison.Accuracy.random-accuracy-CIFAR10 --adjustment_literation 100 --parent_path comparison_experiments --util CIFAR10
```

#### CIFAR-100 Dataset
```shell
# Baseline utility
python -m src.experiments.baseline_comparison.Utility.baseline_utility-CIFAR100 --adjustment_literation -1 --parent_path comparison_experiments --util CIFAR100
# PGI-RDFL utility
python -m src.experiments.baseline_comparison.Utility.pgi_rdfl-utility-CIFAR100 --adjustment_literation -1 --parent_path comparison_experiments --util CIFAR100
# Fixed utility
python -m src.experiments.baseline_comparison.Utility.fixed-utility-CIFAR100 --adjustment_literation -1 --parent_path comparison_experiments --util CIFAR100
# Random utility
python -m src.experiments.baseline_comparison.Utility.random-utility-CIFAR100 --adjustment_literation -1 --parent_path comparison_experiments --util CIFAR100

# Baseline accuracy
python -m src.experiments.baseline_comparison.Accuracy.baseline_accuracy-CIFAR100 --adjustment_literation 100 --parent_path comparison_experiments --util CIFAR100
# PGI-RDFL accuracy
python -m src.experiments.baseline_comparison.Accuracy.pgi_rdfl-accuracy-CIFAR100 --adjustment_literation 100 --parent_path comparison_experiments --util CIFAR100
# Fiexed accuracy
python -m src.experiments.baseline_comparison.Accuracy.fixed-accuracy-CIFAR100 --adjustment_literation 100 --parent_path comparison_experiments --util CIFAR100
# Random accuracy
python -m src.experiments.baseline_comparison.Accuracy.random-accuracy-CIFAR100 --adjustment_literation 100 --parent_path comparison_experiments --util CIFAR100
```

## 🚀 Usage

To run any experiment, use the following command format:
```bash
python -m <module-path> --adjustment_literation <iterations> --parent_path <log-directory> --util <dataset>
```

### Parameters:
- `--adjustment_literation`: Number of iterations for adjustment (-1 for no adjustment, >0 for specific iterations)
- `--parent_path`: Directory for logging results
- `--util`: Dataset to use (MNIST, CIFAR10, CIFAR100)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or support, please open an issue on the GitHub repository.
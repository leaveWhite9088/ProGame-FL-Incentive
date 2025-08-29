📊 Verify Incentives Experiments
Each verify incentives experiment takes about 30 minutes to run.

fixed Strategy

```shell
# MNIST with fixed strategy -utility
python -m src.experiments.verify_incentives.fixed.verify_incentives-fixed-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST

# MNIST with fixed strategy -utility -test
python -m src.experiments.verify_incentives.fixed.verify_incentives-fixed-utility-MNIST-test --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST
```

PGI-RDFL(Our Method)

```shell
# MNIST with PGI-RDFL -utility
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST
# MNIST with PGI-RDFL -accuracy
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_incentives-accuracy --util MNIST

# MNIST with PGI-RDFL -utility -test
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-utility-MNIST-test --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST

```

random Strategy

```shell
# MNIST with random strategy -utility
python -m src.experiments.verify_incentives.random.verify_incentives-random-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility --util MNIST
```

📊 Verify Selection Experiments
Each verify incentives experiment takes about 30 minutes to run.

all_selection

```shell
# MNIST with all_selection -utility
python -m src.experiments.verify_selection.all_selection.verify_selection-as-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# MNIST with all_selection -accuracy
python -m src.experiments.verify_selection.all_selection.verify_selection-as-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy --util MNIST
```

non_two_way_selection(用其结果代替random结果)

```shell
# MNIST with non_two_way_selection -utility
python -m src.experiments.verify_selection.non_two_way_selection.verify_selection-ntws-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# MNIST with non_two_way_selection -accuracy
python -m src.experiments.verify_selection.non_two_way_selection.verify_selection-ntws-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy --util MNIST
```

random_selection

```shell
# MNIST with random_selection -utility
python -m src.experiments.verify_selection.random_selection.verify_selection-rs-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# MNIST with random_selection -accuracy
python -m src.experiments.verify_selection.random_selection.verify_selection-rs-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy --util MNIST
```

two_way_selection

```shell
# MNIST with two_way_selection -utility
python -m src.experiments.verify_selection.two_way_selection.verify_selection-tws-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# MNIST with two_way_selection -accuracy
python -m src.experiments.verify_selection.two_way_selection.verify_selection-tws-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy --util MNIST
```

对比实验
MNIST
```shell
# 基线效用
python -m src.experiments.baseline_comparison.Utility.baseline_utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# PGI-RDFL 效用
python -m src.experiments.baseline_comparison.Utility.pgi_rdfl-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# fixed 效用
python -m src.experiments.baseline_comparison.Utility.fixed-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# random 效用
python -m src.experiments.baseline_comparison.Utility.random-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST

# 基线精度
python -m src.experiments.baseline_comparison.Accuracy.baseline_accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-utility --util MNIST
# PGI-RDFL 精度
python -m src.experiments.baseline_comparison.Accuracy.pgi_rdfl-MNIST --adjustment_literation 100 --parent_path log-verify_selection-utility --util MNIST
# fair_flearn-master 精度

# FL-Client-Sampling-main 精度

# Oort-master 精度
```

CIFRA10
```shell
# 基线效用
python -m src.experiments.baseline_comparison.Utility.baseline_utility-CIFAR10 --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# PGI-RDFL 效用
python -m src.experiments.baseline_comparison.Utility.pgi_rdfl-utility-CIFAR10 --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# fixed 效用
python -m src.experiments.baseline_comparison.Utility.fixed-utility-CIFAR10 --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST
# random 效用
python -m src.experiments.baseline_comparison.Utility.random-utility-CIFAR10 --adjustment_literation -1 --parent_path log-verify_selection-utility --util MNIST

# 基线精度
python -m src.experiments.baseline_comparison.Accuracy.baseline_accuracy-CIFAR10 --adjustment_literation 100 --parent_path log-verify_selection-utility --util MNIST
# PGI-RDFL 精度
python -m src.experiments.baseline_comparison.Accuracy.pgi_rdfl-CIFAR10 --adjustment_literation 100 --parent_path log-verify_selection-utility --util MNIST
# fair_flearn-master 精度

# FL-Client-Sampling-main 精度

# Oort-master 精度
```
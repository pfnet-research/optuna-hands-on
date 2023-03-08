import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST

import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # あればGPUを使う

# データを読み込む (今回はデータが少ないので、全てメモリ上に入れる)
# 訓練用データセット
train = MNIST(root='./data', download=True, train=True)
train_X = train.data.to(device).to(torch.float32) # 60000x28x28
train_y = train.targets.to(device) # 60000

# 評価用データセット
validation = MNIST(root='./data', download=True, train=False)
validation_X = validation.data.to(device).to(torch.float32) # 10000x28x28
validation_y = validation.targets.to(device) # 10000

def fit_mnist_and_evaluate_with_report(trial, model, optimizer):

    # 損失関数の定義
    loss_func = nn.CrossEntropyLoss() 

    # 学習 (今回はepoch数は20で固定とします。)
    epochs = 20
    for epoch in range(epochs):
        
        loss_sum = 0.0
        
        # batchをシャッフルする (今回はbatch sizeは600で固定とします。)
        batch_size = 600
        batch_idxs = torch.randperm(len(train_X), device=device).view(-1, batch_size)

        for i, batch in enumerate(batch_idxs):
            # 各batchについて最適化を回す

            optimizer.zero_grad() # 微分係数の初期化
            outputs = model(train_X[batch])          # 予測
            loss = loss_func(outputs, train_y[batch]) # 損失関数の計算 
            loss.backward()  # 微分の計算
            optimizer.step() # 最適化の1ステップの計算 

            loss_sum += loss.item()

        train_loss = loss_sum / (i + 1) # batchごとの損失関数の平均をとる

        # 評価
        with torch.no_grad():
            outputs = model(validation_X)
            _, predicted = torch.max(outputs.data, dim=1) # 最も予測値が高かったものをとる
            total = len(validation_X)
            correct = (predicted == validation_y).sum().item()
            validation_accuracy = correct / total
        
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.3f}, validation_accuracy={validation_accuracy:.4f}")

        # 中間値をOptunaに報告 <-- この3行が追加されている
        trial.report(validation_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return validation_accuracy

def objective_variable_depth(trial):
        
    # モデルの定義
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 5)  # 隠れ層の数
    activation_funcs = {
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'ReLU': nn.ReLU(),
        'ELU': nn.ELU(),
    }

    model = nn.Sequential()
    model.add_module('flatten', nn.Flatten()) # 二次元の画像を一次元に変換
    last_dim = 28 * 28 # 最初の入力層の次元数
    for i in range(n_hidden_layers):
        hidden_dim = trial.suggest_int(f'hidden_dim[{i}]', 10, 50) # 中間層の次元数
        activation_func = trial.suggest_categorical(f'activation_func[{i}]', ['Sigmoid', 'Tanh', 'ReLU', 'ELU']) # 中間層の活性化関数
        activation = activation_funcs[activation_func]

        model.add_module(f'linear[{i}]', nn.Linear(last_dim, hidden_dim)) # 線形変換
        model.add_module(f'activation[{i}]', activation) # 活性化関数
        last_dim = hidden_dim
    
    model.add_module(f'linear[{n_hidden_layers}]', nn.Linear(last_dim, 10)) # 出力層の線形変換

    model = model.to(device)

    # 最適化アルゴリズムの定義
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True) # 学習率
    momentum = trial.suggest_float("momentum", 0.5, 1.0) # モーメンタム
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

    validation_accuracy = fit_mnist_and_evaluate_with_report(trial, model, optimizer)
    return validation_accuracy

study_variable_depth = optuna.create_study(
    direction="maximize", 
    sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True), 
    pruner=optuna.pruners.HyperbandPruner(),
    storage=optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("./mnist_study.optuna")),  # 保存するファイルパスを指定
    study_name="distributed_study", # Study名を指定。既存のStudy名と同一のものを指定することで、同じStudyを参照できる。
    load_if_exists=True)
study_variable_depth.optimize(objective_variable_depth, n_trials=int(sys.argv[1]))

print(f"最良の精度: {study_variable_depth.best_value}")
print(f"最良のハイパーパラメータ: {study_variable_depth.best_params}")
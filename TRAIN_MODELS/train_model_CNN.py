import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # ciul wie co z tym błędem potem na necie poszukać czemu to naprawia jakiś błąd -_-

import torch
import torch.nn as nn
import torch.optim as optim

import torchmetrics

from torchvision.transforms import v2

from model_CNN import TinyVGG_01
from engine import train
from utils import save_model
from dataSetup import create_data_loader
from tenserflow import create_writer

# import matplotlib.pyplot as plt
# import pandas as pd
# from torchinfo import summary


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Miejsce zapisu testów modeli. W następujący sposób:
    # - logs/YYYY-MM-DD/experiment_name/model_name/extra
    # - logs/YYYY-MM-DD/experiment_name/model_name      (bez extra)
    experiment_name = "test1"
    model_name = "TinyVGG_01"
    extra = "czysty"

    # Zapisuje testy dla tensorboard
    # writer = create_writer(experiment_name=experiment_name,
    #                        model_name=model_name,
    #                        extra=extra)

    # Hiperparametry
    NUM_WORKERS = 8 # os.cpu_count()
    BATCH_SIZE = 128
    HIDDEN_UNITS = 200
    LEARNING_RATE = 2e-05
    GAMMA = 0.9 # czyli lr zostanie przez to pomnożone
    L2_WEIGHT_DECAY = 3e-5
    EPOCHS = 4

    SAVE_MODEL = False
    LOAD_MODEL = False


    transform = v2.Compose([
        v2.Resize(size=(48, 48)),
        v2.ToImage(),
        v2.AutoAugment(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # Ścieżki danych treningowych i testowych
    train_path = "../dane/train"
    test_path = "../dane/test"

    # Ścieżki modeli oraz nazwa modelu
    target_dir = "../models"
    model_name_file = "tinyvgg_model_01.pth"
    path_model = os.path.join(target_dir, model_name_file)


    train_data_loader, test_data_loader, class_names = create_data_loader(train_path, test_path, transform, BATCH_SIZE, NUM_WORKERS)



    model_01 = TinyVGG_01(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)

    # summary(model=model_01, input_size=(1, 3, 48, 48),col_names=["input_size","output_size","num_params","kernel_size","trainable",], device="cuda") # podsumowuje model pod względem parametrów i innych zmiennych
    # train_features, train_labels = next(iter(train_data_loader))
    # print("train_features",train_features.shape)
    # print(train_features[0])

    if LOAD_MODEL:
        model_01.load_state_dict(torch.load(path_model, weights_only=True))
        print(f"Załadowany model {model_name}")

    # Funckje potrzebne do trenowania modeli
    accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model_01.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=L2_WEIGHT_DECAY)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    # set a scheduler to decay the learning rate by 0.1 on the 100th 150th epochs
    #lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)
    # Trenowanie modelu
    results = train(model=model_01,
          train_dataloader=train_data_loader,
          test_dataloader=test_data_loader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          lr_schedule=lr_schedule,
          accuracy_fn=accuracy_fn,
          epochs=EPOCHS,
          device=device
          )
    # writer.close()

    if SAVE_MODEL:
        save_model(model=model_01,
                   target_dir=target_dir,
                   model_name=model_name_file)
        print(f"Model zostały zapisany {model_name_file}, w {target_dir}")
if __name__ == '__main__':
    main()


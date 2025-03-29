from tqdm.auto import tqdm
import torch
import torchmetrics
from typing import List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               lr_schedule: torch.optim.lr_scheduler,
               accuracy_fn: torchmetrics.classification,
               device: torch.device) -> Tuple[float, float]:
    train_loss, train_acc = 0, 0

    model.train()

    for batch, (img, y_true) in enumerate(dataloader):
        img, y_true = img.to(device), y_true.to(device)

        y_pred = model(img) # wyglądają na znormalizowane [0.0235, 0.0235, 0.0235,  ..., 0.0196, 0.0196, 0.0196]
        loss = loss_fn(y_pred, y_true)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = accuracy_fn(y_pred, y_true)
        train_acc += accuracy.item()
    lr_schedule.step()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn: torchmetrics.classification,
               device: torch.device) -> Tuple[float, float]:
    test_loss, test_acc = 0, 0

    model.eval()

    with torch.no_grad():
        for batch, (img, y_true) in enumerate(dataloader):
            img, y_true = img.to(device), y_true.to(device)
            y_pred = model(img)
            loss = loss_fn(y_pred, y_true)
            test_loss += loss

            accuracy = accuracy_fn(y_pred, y_true)
            test_acc += accuracy.item()

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          lr_schedule: torch.optim.lr_scheduler,
          accuracy_fn: torchmetrics.classification,
          epochs: int,
          device: torch.device,
          writer = None) -> dict[str,List]:

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           lr_schedule=lr_schedule,
                                           accuracy_fn=accuracy_fn,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device)
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={
                                   "train_loss": train_loss,
                                   "test_loss": test_loss
                               },
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={
                                   "train_acc": train_acc,
                                   "test_acc": test_acc
                               },
                               global_step=epoch)
            writer.add_graph(model=model,
                             input_to_model=torch.randn(64,1,128,128).to(device))

    return results
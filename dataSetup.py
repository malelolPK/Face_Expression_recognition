from torchvision import datasets
from torch.utils.data import DataLoader

def create_data_loader(train_dir, test_dir, transform, BATCH_SIZE, NUM_WORKERS):

    train_datasets = datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )

    test_datasets = datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )

    classes = test_datasets.classes

    train_dataloader = DataLoader(
        dataset=train_datasets,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,  # Workerzy pozostają aktywni między epokami
        prefetch_factor=2  # Każdy worker ładuje kilka partii z wyprzedzeniem
    )

    test_dataloader = DataLoader(
        dataset=test_datasets,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,  # Workerzy pozostają aktywni między epokami
        prefetch_factor=2  # Każdy worker ładuje kilka partii z wyprzedzeniem
    )

    return train_dataloader, test_dataloader, classes
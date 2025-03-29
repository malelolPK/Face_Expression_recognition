import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None):
    """
    Tworzy i zwraca obiekt SummaryWriter dla TensorBoard.

    Logi są zapisywane w strukturze katalogów:
      logs/YYYY-MM-DD/experiment_name/model_name/[extra]

    Argumenty:
      experiment_name (str): Nazwa eksperymentu.
      model_name (str): Nazwa modelu.
      extra (str, opcjonalnie): Dodatkowy podkatalog, który umożliwia dalsze
                                rozdzielenie logów (np. wersja eksperymentu).

    :return: obiekt SummaryWriter

    Przykład użycia:
      writer = create_writer("eksperyment1", "modelA", "v1")
    """
    # Ustawia 
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("logs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("logs", timestamp, experiment_name, model_name)

    print(f"SummaryWritter saving to {log_dir}")
    return SummaryWriter(log_dir=log_dir)
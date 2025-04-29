""" Initiates the training of the classifier specified in the constants file. """
import datetime
from typing import Tuple
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
import torch
from tqdm import tqdm
import src.constants as const
from src import utils
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
from src.data_processing import SDDataset
from torch_geometric.data import Data
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        # pred[pred < 0] = 0
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


class WeightedMSELoss(torch.nn.Module):
    """
    Weights the MSE using a vector of weights for each sample.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, weight):
        return (weight * (pred - actual) ** 2).sum()


def subsampleClasses(
    y: torch.Tensor, y_hat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # subsample the majority class
    source_indicator = 0 if const.MODEL == "GCNR" else 1
    non_sources = torch.where(y != source_indicator)[0]
    sources = torch.where(y == source_indicator)[0]
    random_numbers = torch.randperm(non_sources.shape[0])[: sources.shape[0]]
    subsampled_non_sources = non_sources[random_numbers]
    indices = torch.cat((subsampled_non_sources, sources))
    return y[indices], y_hat[indices]


def node_weights(y: torch.Tensor) -> torch.Tensor:
    """
    Calculates the weights for each sample in the batch.
    :param y: The labels of the batch.
    :return: The weights for each sample in the batch.
    """
    source_indicator = 0 if const.MODEL == "GCNR" else 1
    non_sources = torch.where(y != source_indicator)[0]
    sources = torch.where(y == source_indicator)[0]
    weights = torch.ones(y.shape[0])
    weights[non_sources] = 1
    weights[sources] = 1 / sources.shape[0] * non_sources.shape[0]
    return weights


def graph_weights(data_list: list[Data]) -> torch.Tensor:
    """
    Calculates node weights based on the size of the graph.
    Each nodes weight is 1 / number of nodes in the graph.
    :param data_list: The list of graphs.
    :return: A vector of weights containing a weight for each node.
    """
    weights = []
    for data in data_list:
        weights.extend([1 / data.num_nodes] * data.num_nodes)
    return torch.Tensor(weights)


def configure_model_and_loader(
    model: torch.nn.Module, dataset: SDDataset
) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader, bool]:
    """
    Configures the model and data loader based on the available devices (GPU/CPU).
    :param model: The model to configure.
    :param dataset: The dataset to use for training.
    :return: A tuple containing the configured model, data loader, and a flag indicating if DataParallel is used.
    """
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model).to(device)
        loader = DataListLoader(dataset, batch_size=const.BATCH_SIZE, shuffle=True)
        is_data_parallel = True
    else:
        print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=const.BATCH_SIZE, shuffle=True)
        is_data_parallel = False

    return model, loader, is_data_parallel


def extract_labels(data_list, is_data_parallel: bool, device: torch.device) -> torch.Tensor:
    """
    Extracts the labels from the data depending on the input type.
    :param data_list: The batch of data, either as a list (DataListLoader) or a single batch (DataLoader).
    :param is_data_parallel: Whether the data is being used in a DataParallel setting.
    :param device: The device to move the labels to.
    :return: A tensor of labels.
    """
    if is_data_parallel:
        return torch.cat([data.y for data in data_list]).to(device)
    return data_list.y.to(device)


def save_model_checkpoint(model: torch.nn.Module, model_name: str, is_data_parallel: bool):
    """
    Saves the model checkpoint.
    :param model: The model to save.
    :param model_name: The name of the model.
    :param is_data_parallel: Whether the model is wrapped in DataParallel.
    """
    if is_data_parallel:
        utils.save_model(model.module, "latest")
        utils.save_model(model.module, model_name)
    else:
        utils.save_model(model, "latest")
        utils.save_model(model, model_name)


def train(
    model: torch.nn.Module,
    model_name: str,
    dataset: SDDataset,
    criterion: torch.nn.Module,
):
    """
    Trains the specified model using the given dataset.
    :param model: The model to train.
    :param model_name: The name of the model.
    :param dataset: The dataset to train on, containing the graph structure, features, and labels.
    :param criterion: The loss criterion used for training.
    :return: The trained model.
    """
    # Configure model and loader
    model, loader, is_data_parallel = configure_model_and_loader(model, dataset)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=const.WEIGHT_DECAY)
    epochs = range(1, const.EPOCHS)
    print(f"Train Model on device: {device}")
    min_loss = float("inf")

    for epoch in tqdm(epochs, disable=const.ON_CLUSTER):
        agg_loss = 0
        for data_list in loader:
            optimizer.zero_grad()
            out = model(data_list)

            y = extract_labels(data_list, is_data_parallel, out.device)

            if const.SUBSAMPLE:
                y, out = subsampleClasses(y, out)
            w = torch.ones(y.shape[0])
            if const.CLASS_WEIGHTING:
                w = node_weights(y).to(out.device)
            if const.GRAPH_WEIGHTING:
                w *= graph_weights(data_list).to(out.device)
            if const.GRAPH_WEIGHTING or const.CLASS_WEIGHTING:
                loss = criterion(out, y, w)
            else:
                loss = criterion(out, y)

            loss.backward()
            optimizer.step()
            agg_loss += loss.item()

        print(f"Epoch: {epoch}\tLoss: {agg_loss:.4f}")
        writer.add_scalar("Loss/train", agg_loss, epoch)
        if agg_loss < min_loss:
            print("Saving new best model ...")
            min_loss = agg_loss
        save_model_checkpoint(model, model_name, is_data_parallel)

    writer.flush()
    writer.close()
    return model



def main():
    """
    Initiates the training of the classifier specified in the constants file.
    """
    print("Prepare Data ...")

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_name = (
        f"{const.MODEL}_{current_time}"
        if const.MODEL_NAME is None
        else const.MODEL_NAME
    )

    if const.MODEL == "GCNR":
        model = GCNR()
        criterion = MSLELoss() if const.USE_LOG_LOSS else torch.nn.MSELoss()
        if const.CLASS_WEIGHTING or const.GRAPH_WEIGHTING:
            criterion = WeightedMSELoss()

    elif const.MODEL == "GCNSI":
        model = GCNSI()
        criterion = torch.nn.BCEWithLogitsLoss()

    train_data = utils.load_processed_data(validation=False)
    train(model, model_name, train_data, criterion)


if __name__ == "__main__":
    main()

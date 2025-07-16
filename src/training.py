import datetime
from typing import Tuple
from architectures.GCNSI import GCNSI
from architectures.GAT import GAT
import torch
from tqdm import tqdm
import src.constants as const
from src import utils
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
from src.data_processing import SDDataset
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, weight):
        return (weight * (pred - actual) ** 2).sum()


def subsampleClasses(y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    source_indicator = 1
    non_sources = torch.where(y != source_indicator)[0]
    sources = torch.where(y == source_indicator)[0]
    random_numbers = torch.randperm(non_sources.shape[0])[: sources.shape[0]]
    subsampled_non_sources = non_sources[random_numbers]
    indices = torch.cat((subsampled_non_sources, sources))
    return y[indices], y_hat[indices]


def node_weights(y: torch.Tensor) -> torch.Tensor:
    source_indicator = 1
    non_sources = torch.where(y != source_indicator)[0]
    sources = torch.where(y == source_indicator)[0]
    weights = torch.ones(y.shape[0])
    weights[sources] = 1 / sources.shape[0] * non_sources.shape[0]
    return weights


def graph_weights(data_list: list[Data]) -> torch.Tensor:
    weights = []
    for data in data_list:
        weights.extend([1 / data.num_nodes] * data.num_nodes)
    return torch.Tensor(weights)


def configure_model_and_loader(model: torch.nn.Module, train_dataset: SDDataset, val_dataset: SDDataset) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader, bool]:
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model).to(device)
        train_loader = DataListLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True)
        val_loader = DataListLoader(val_dataset, batch_size=const.BATCH_SIZE, shuffle=False)
        is_data_parallel = True
    else:
        print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")
        model = model.to(device)
        train_loader = DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=const.BATCH_SIZE, shuffle=False)
        is_data_parallel = False
    return model, train_loader, val_loader, is_data_parallel


def extract_labels(data_list, is_data_parallel: bool, device: torch.device) -> torch.Tensor:
    if is_data_parallel:
        return torch.cat([data.y for data in data_list]).to(device)
    return data_list.y.to(device)


def save_model_checkpoint(model: torch.nn.Module, model_name: str, is_data_parallel: bool):
    if is_data_parallel:
        utils.save_model(model.module, "latest")
        utils.save_model(model.module, model_name)
    else:
        utils.save_model(model, "latest")
        utils.save_model(model, model_name)


@torch.no_grad()
def validate(model, val_loader, criterion, is_data_parallel):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data_list in val_loader:
        if not is_data_parallel:
            data_list = data_list.to(device)

        out = model(data_list)
        y = extract_labels(data_list, is_data_parallel, out.device)
        y = y.view(-1, 1).float()

        loss = criterion(out, y)
        preds = (out.sigmoid() > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    acc = correct / total if total > 0 else None

    return avg_loss, acc



def train(model: torch.nn.Module, model_name: str, train_dataset: SDDataset, val_dataset: SDDataset, criterion: torch.nn.Module):
    model, train_loader, val_loader, is_data_parallel = configure_model_and_loader(model, train_dataset, val_dataset)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE, weight_decay=const.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    early_stop_patience = 10
    best_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(1, const.EPOCHS + 1), disable=const.ON_CLUSTER):
        model.train()
        agg_train_loss = 0
        batch_count = 0

        for data_list in train_loader:
            optimizer.zero_grad()
            if not is_data_parallel:
                data_list = data_list.to(device)

            out = model(data_list)
            y = extract_labels(data_list, is_data_parallel, out.device)
            y = y.view(-1, 1).float()  # match shape + dtype

            if const.SUBSAMPLE:
                y, out = subsampleClasses(y, out)

            w = torch.ones(y.shape[0]).to(out.device)
            if const.CLASS_WEIGHTING:
                w = node_weights(y).to(out.device)
                if epoch == 1 and batch_count == 0:
                    print(f"Weights range: [{w.min():.3f}, {w.max():.3f}]")
                    print(f"Weights for positive samples: {w[y.flatten() == 1]}")
                    
            if const.GRAPH_WEIGHTING:
                w *= graph_weights(data_list).to(out.device)

            if const.CLASS_WEIGHTING or const.GRAPH_WEIGHTING:
                loss = criterion(out, y, w)
            else:
                loss = criterion(out, y)
                
            if epoch == 1 and batch_count == 0:
                print(f"Loss: {loss.item():.6f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            agg_train_loss += loss.item()
            batch_count += 1

        writer.add_scalar("Loss/train", agg_train_loss, epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        val_loss, val_acc = validate(model, val_loader, criterion, is_data_parallel)
        writer.add_scalar("Loss/val", val_loss, epoch)
        if val_acc is not None:
            writer.add_scalar("Accuracy/val", val_acc, epoch)
        print(f"Epoch {epoch}: Train Loss = {agg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        if val_loss < best_loss:
            print("Saving new best model ...")
            best_loss = val_loss
            patience_counter = 0
            save_model_checkpoint(model, model_name, is_data_parallel)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    writer.flush()
    writer.close()
    return model


def main():
    print("Prepare Data ...")
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_name = f"{const.MODEL}_{current_time}" if const.MODEL_NAME is None else const.MODEL_NAME
    
    if const.MODEL == "GCNSI":
        model = GCNSI()
        criterion = torch.nn.BCEWithLogitsLoss()
    elif const.MODEL == "GAT":
        model = GAT()
        # Use weighted BCE loss for GAT to handle class imbalance
        # Calculate pos_weight based on expected class distribution
        # Assuming roughly 1 source per graph with ~10-20 nodes
        pos_weight = torch.tensor([10.0]).to(device)  # Give 10x weight to positive class, move to device
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_data = utils.load_processed_data(split="train")
    val_data = utils.load_processed_data(split="val")
    train(model, model_name, train_data, val_data, criterion)


if __name__ == "__main__":
    main()

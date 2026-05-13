import pickle
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


def make_data_loaders(train_data, val_data, test_data, batch_size=20):
    train_loader = DataLoader(train_data, batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader

def sparsity_loss(A, lambda_sparse=1e-5):
    return lambda_sparse * torch.mean(torch.abs(A))

def train_lower_level(gcn, generator, loader, optimizer, device):
    gcn.train()
    generator.eval()
    for batch in loader:
        batch = batch.to(device)
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            x_i = batch.x[mask]
            y_i = batch.y[i].unsqueeze(0)
            A_i = generator(x_i)
            optimizer.zero_grad()
            logits = gcn(x_i, A_i)
            loss = F.cross_entropy(logits, y_i)
            loss.backward()
            optimizer.step()


def train_upper_level(
    gcn, generator,
    loader, optimizer,
    device, lambda_sparse=1e-5,
    beta_kl=0.0,
):
    gcn.eval()
    generator.train()

    for batch in loader:
        batch = batch.to(device)

        for i in range(batch.num_graphs):
            mask = batch.batch == i

            x_i = batch.x[mask]
            y_i = batch.y[i].unsqueeze(0)

            optimizer.zero_grad()

            A_i, kl = generator(x_i, return_kl=True)
            logits = gcn(x_i, A_i)

            loss = (
                F.cross_entropy(logits, y_i)
                + sparsity_loss(A_i, lambda_sparse)
                + beta_kl * kl
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()

@torch.no_grad()
def compute_lower_loss(gcn, generator, loader, device):
    gcn.eval()
    generator.eval()

    total_loss = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        for i in range(batch.num_graphs):
            mask = batch.batch == i

            x_i = batch.x[mask]
            y_i = batch.y[i].unsqueeze(0)

            A_i = generator(x_i)
            logits = gcn(x_i, A_i)

            loss = F.cross_entropy(logits, y_i)

            total_loss += float(loss.item())
            n_graphs += 1

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def compute_upper_loss(
    gcn, generator, loader, device, lambda_sparse=1e-5, beta_kl=0.0,
):
    gcn.eval()
    generator.eval()

    total_loss = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        for i in range(batch.num_graphs):
            mask = batch.batch == i
            x_i = batch.x[mask]
            y_i = batch.y[i].unsqueeze(0)
            A_i, kl = generator(x_i, return_kl=True)
            logits = gcn(x_i, A_i)
            # we set beta_kl to 0, no use of kl divengence. You can use if you like to.
            loss = (
                F.cross_entropy(logits, y_i)
                + sparsity_loss(A_i, lambda_sparse)
                + beta_kl * kl
            )
            total_loss += float(loss.item())
            n_graphs += 1

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate_accuracy(gcn, generator, loader, device):
    gcn.eval()
    generator.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for batch in loader:
        batch = batch.to(device)
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            x_i = batch.x[mask]
            y_i = batch.y[i].unsqueeze(0)
            A_i = generator(x_i)
            logits = gcn(x_i, A_i)
            pred = logits.argmax(dim=1)
            correct += int(pred.item() == y_i.item())
            total += 1
            y_true.append(int(y_i.item()))
            y_pred.append(int(pred.item()))
    acc = correct / total if total > 0 else 0.0
    return acc, y_true, y_pred


def bilevel_train(
    gcn, generator,
    train_loader, val_loader, test_loader,
    lr_gcn=1e-4, lr_gen=1e-3,
    num_iterations=200, lower_epochs=1, upper_epochs=1, lambda_sparse=1e-5,
    beta_kl=0.0, tau_init=1.0, tau_min=0.1, tau_decay=0.98, device="cpu",
):
    gcn.to(device)
    generator.to(device)
    optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=lr_gcn)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=lr_gen)
    generator.tau = tau_init
    history = {
        "iter": [],
        "lower_loss": [],
        "upper_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "tau": [],
    }

    for it in range(num_iterations):
        generator.tau = max(tau_min, generator.tau * tau_decay)
        for _ in range(lower_epochs):
            train_lower_level(
                gcn, generator, train_loader, optimizer_gcn,
                device,
            )
        for _ in range(upper_epochs):
            train_upper_level(
                gcn, generator, val_loader,
                optimizer_gen,
                device,
                lambda_sparse=lambda_sparse,
                beta_kl=beta_kl,
            )
        lower_loss = compute_lower_loss(
            gcn, generator,
            train_loader,
            device,
        )
        upper_loss = compute_upper_loss(
            gcn, generator,
            val_loader,
            device,
            lambda_sparse=lambda_sparse,
            beta_kl=beta_kl,
        )
        train_acc, _, _ = evaluate_accuracy(
            gcn, generator,
            train_loader,
            device,
        )
        val_acc, _, _ = evaluate_accuracy(
            gcn, generator,
            val_loader,
            device,
        )
        test_acc, _, _ = evaluate_accuracy(
            gcn, generator,
            test_loader,
            device,
        )
        print(
            f"Iter {it + 1:03d}/{num_iterations} | "
            f"Lower {lower_loss:.4f} | "
            f"Upper {upper_loss:.4f} | "
            f"Train {train_acc * 100:.2f}% | "
            f"Val {val_acc * 100:.2f}% | "
            f"Test {test_acc * 100:.2f}% | "
            f"tau {generator.tau:.4f}"
        )
        history["iter"].append(it + 1)
        history["lower_loss"].append(lower_loss)
        history["upper_loss"].append(upper_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["tau"].append(float(generator.tau))
    return gcn, generator, history

def save_checkpoint(path, gcn, generator, history=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "gcn_state": gcn.state_dict(),
        "generator_state": generator.state_dict(),
    }
    if history is not None:
        checkpoint["history"] = history
    torch.save(checkpoint, path)

def save_history(path, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(history, f)
import argparse
from pathlib import Path
import torch
from abignet_data import (
    set_seed, prepare_train_val_test_data,
)
from abignet_model import (
    DenseGCN, AdjacencyGenerator,
)
from abignet_train import (
    make_data_loaders,
    bilevel_train,
    evaluate_accuracy,
    save_checkpoint,
    save_history,
)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dirs", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-dirs", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--test-dirs", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--out-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr-gcn", type=float, default=1e-4)
    parser.add_argument("--lr-gen", type=float, default=1e-3)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--lower-epochs", type=int, default=1)
    parser.add_argument("--upper-epochs", type=int, default=1)
    parser.add_argument("--lambda-sparse", type=float, default=1e-5)
    parser.add_argument("--beta-kl", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-dir", default="outputs/img001_img002_img003")
    args = parser.parse_args()
    set_seed(args.seed)
    train_data, val_data, test_data, scaler = prepare_train_val_test_data(
        train_dirs=args.train_dirs,
        train_labels=args.train_labels,
        val_dirs=args.val_dirs,
        val_labels=args.val_labels,
        test_dirs=args.test_dirs,
        test_labels=args.test_labels,
    )

    train_loader, val_loader, test_loader = make_data_loaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size,
    )

    in_dim = train_data[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Input feature dimension:", in_dim)
    gcn = DenseGCN(
        in_dim=in_dim, out_dim=args.out_dim,
    )
    generator = AdjacencyGenerator(
        in_dim=in_dim,
        hidden_dim1=32,
        hidden_dim2=16,
    )

    gcn, generator, history = bilevel_train(
        gcn=gcn,
        generator=generator,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr_gcn=args.lr_gcn,
        lr_gen=args.lr_gen,
        num_iterations=args.num_iterations,
        lower_epochs=args.lower_epochs,
        upper_epochs=args.upper_epochs,
        lambda_sparse=args.lambda_sparse,
        beta_kl=args.beta_kl,
        device=device,
    )

    test_acc, y_true, y_pred = evaluate_accuracy(
        gcn, generator,
        test_loader,
        device,
    )

    print(f"\nFinal test accuracy: {test_acc * 100:.2f}%")
    print("True labels:", y_true)
    print("Pred labels:", y_pred)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "model_checkpoint.pth",
        gcn, generator,
        history,
    )
    save_history(
        output_dir / "history.pkl",
        history,
    )
    print("Saved outputs to:", output_dir)

if __name__ == "__main__":
    main()
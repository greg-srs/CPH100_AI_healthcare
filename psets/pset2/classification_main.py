"""
Main script for PathMNIST classification (Part 1).
Students should implement the TODO sections to achieve >90% accuracy.
"""

import sys
import os

from classification_dataset import create_pathmnist_dataloaders
from classification_models import get_model, count_parameters
from classification_train import train_model
from plot_utils import save_history, plot_history

import torch
torch.manual_seed()

def main(args):
    print("=== CPH 100A Problem Set 2 - Part 1: PathMNIST Classification ===")
    
    # Load data
    print("Loading PathMNIST dataset...")
    train_loader, val_loader, num_classes = create_pathmnist_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
        data_root='./data'
    )
    
    print(f"\nCreating {args.model_name} model...")
    model = get_model(args.model_name, num_classes=num_classes)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    print(f"\nTraining {args.model_name} model...")
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps_per_epoch=args.max_steps_per_epoch  # Fast exploration mode. #TODO: Change for your full runs
        )
        
        print("Training completed successfully!")

        # Save the per-epoch history and the learning curves for the report
        run_name = f"{args.model_name}_{args.num_epochs}ep_lr{args.learning_rate}_wd{args.weight_decay}"
        save_history(history, f"report/figures/{run_name}_history.json")
        plot_history(history, f"report/figures/{run_name}_curves.png",
                     title=f"{args.model_name} - {args.num_epochs} epochs, lr={args.learning_rate}, wd={args.weight_decay}")

        # Show final results
        if history['val_acc']:
            best_val_acc = max(history['val_acc'])
            print(f"Best validation accuracy: {best_val_acc:.4f}")
        
    except NotImplementedError as e:
        print(f"❌ Training failed: {e}")
        return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPH 100A Problem Set 2 - PathMNIST Classification')
    parser.add_argument('--model_name', type=str, default='mlp',
                       choices=['mlp', 'cnn1','cnn2', 'cnn3','linear'], #TODO: add your models names here
                       help='Model to train')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps_per_epoch', type=int, default=100)
    args = parser.parse_args()
    
    main(args) 
"""
Main script for PathMNIST box segmentation (Part 2).
U-Net segmentation of colored boxes on PathMNIST images.
"""

import sys
import os
import torch
import torchvision.transforms as transforms

from synthetic_data import create_box_dataloaders, visualize_box_samples, PathMNISTBoxDataset
from segmentation_models import get_segmentation_model, count_parameters
from segmentation_train import train_segmentation_model, evaluate_segmentation_model, visualize_predictions

from plot_utils import save_history, plot_history

def main(args):
    print("=== CPH 100A Problem Set 2 - Part 2: Box Segmentation ===")
    
    
    # Create box datasets
    print("Creating box segmentation datasets...")
    
    # Set seed for reproducible dataset generation
    import torch
    torch.manual_seed(42)
    import random
    random.seed()
    
    train_loader, val_loader, test_loader, num_classes = create_box_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
        data_root='./data'
    )
    
    # Debug: Check dataset characteristics
    print("\nDataset Analysis:")
    val_total = sum(len(batch['has_box']) for batch in val_loader)
    print(f"Validation: {val_total} samples (all have boxes)")
    
    test_total = sum(len(batch['has_box']) for batch in test_loader)
    print(f"Test: {test_total} samples (all have boxes)")
    

    
    # Visualize some samples
    print("\nVisualizing box samples...")
    # Same normalization as the train/val loaders, so the model sees the same input range
    sample_dataset = PathMNISTBoxDataset(
        split='train',
        add_boxes=True,
        transform=transforms.Normalize(mean=[0.5], std=[0.5])
    )
    visualize_box_samples(sample_dataset, num_samples=6, save_path='visualizations/box_samples.png')
    
    # Train the specified model
    print(f"\n{'='*60}")
    print(f"Training {args.model_name.upper()} Segmentation Model")
    print(f"{'='*60}")
    
    # Create model
    model = get_segmentation_model(args.model_name, in_channels=3, out_channels=1)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Train model
    print(f"\nTraining {args.model_name} model...")
    history = train_segmentation_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps_per_epoch=args.max_steps_per_epoch  # Fast exploration mode
    )
    
    # Evaluate model on validation set
    print(f"\nEvaluating {args.model_name} model...")
    val_results = evaluate_segmentation_model(model, val_loader, f"{args.model_name.upper()} Validation")

    # Save the per-epoch history and the learning curves for the report
    run_name = f"seg_{args.model_name}_{args.num_epochs}ep_lr{args.learning_rate}_wd{args.weight_decay}"
    save_history(history, f"report/figures/{run_name}_history.json")
    plot_history(history, f"report/figures/{run_name}_curves.png",
                    title=f"{args.model_name} - {args.num_epochs} epochs, lr={args.learning_rate}, wd={args.weight_decay}")


    
    # Show results
    val_iou = val_results['mean_iou']
    print(f"\n📊 {args.model_name.upper()} Results:")
    print(f"   Final Validation IoU: {val_iou:.4f}")
    
    # Visualize predictions
    print(f"\nVisualizing {args.model_name} predictions...")
    visualize_predictions(
        model, sample_dataset, 
        num_samples=6, 
        save_path=f'visualizations/{args.model_name}_predictions.png'
    )
    
    return {'model': model, 'val_results': val_results}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PathMNIST Black Box Segmentation')
    parser.add_argument('--model_name', type=str, default='unet',
                       choices=['mlp', 'unet', 'linear','cnn'], #TODO: add your models names here
                       help='Segmentation model to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for regularization')
    parser.add_argument('--batch_size', type=int, default=32,
                           help='batch size')
    parser.add_argument('--max_steps_per_epoch', type=int, default=100,
                           help='max_steps_per_epoch ')
    args = parser.parse_args()
    
    
    # Just create and visualize the dataset
    print("Creating and visualizing box dataset...")
    dataset = PathMNISTBoxDataset(
        split='train',
        add_boxes=True,
        transform=transforms.Normalize(mean=[0.5], std=[0.5])
    )
    visualize_box_samples(dataset, num_samples=6, save_path='visualizations/box_samples.png')
    print("Visualization complete!")
    
    results = main(args)
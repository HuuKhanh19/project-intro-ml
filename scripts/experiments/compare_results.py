"""
Compare results from all experiments.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import print_header, print_colored, Colors

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_experiment_results(checkpoint_dir='checkpoints'):
    """Load results from all experiments."""
    checkpoint_dir = Path(checkpoint_dir)
    
    experiments = [
        'exp01_densenet121_weighted_ce',
        'exp02_densenet121_focal',
        'exp03_efficientnet_b0_weighted_ce',
        'exp04_efficientnet_b0_focal',
        'exp05_lenet_weighted_ce',
        'exp06_lenet_focal',
        'exp07_mlp_weighted_ce',
        'exp08_mlp_focal',
    ]
    
    results = []
    
    for exp_id in experiments:
        exp_dir = checkpoint_dir / exp_id
        metrics_file = exp_dir / 'evaluation' / 'metrics_test.json'
        
        if not metrics_file.exists():
            print_colored(f"⚠ Missing results for {exp_id}", Colors.BRIGHT_YELLOW)
            continue
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Parse experiment name
        parts = exp_id.split('_')
        if 'densenet121' in exp_id:
            model = 'DenseNet-121'
        elif 'efficientnet' in exp_id:
            model = 'EfficientNet-B0'
        elif 'lenet' in exp_id:
            model = 'LeNet'
        elif 'mlp' in exp_id:
            model = 'MLP'
        else:
            model = 'Unknown'
        
        loss = 'Weighted CE' if 'weighted_ce' in exp_id else 'Focal Loss'
        
        results.append({
            'experiment_id': exp_id,
            'model': model,
            'loss': loss,
            'accuracy': metrics['accuracy'] * 100,
            'precision': metrics['precision'] * 100,
            'recall': metrics['recall'] * 100,
            'f1': metrics['f1'] * 100,
            'auc': metrics.get('auc', 0) * 100 if metrics.get('auc') else None
        })
    
    return pd.DataFrame(results)


def plot_comparison(df, save_dir='results'):
    """Plot comparison of all experiments."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(df))
    width = 0.35
    
    models = df['model'].tolist()
    losses = df['loss'].tolist()
    labels = [f"{m}\n{l}" for m, l in zip(models, losses)]
    
    bars = ax.bar(x, df['accuracy'], width, label='Accuracy', color='steelblue')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model + Loss', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Comparison Across All Experiments', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: All metrics comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = range(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar([p + offset for p in x], df[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['experiment_id'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Model comparison (grouped by loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Weighted CE
    df_wce = df[df['loss'] == 'Weighted CE'].sort_values('accuracy', ascending=False)
    ax1.barh(df_wce['model'], df_wce['accuracy'], color='skyblue')
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Weighted Cross-Entropy Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (model, acc) in enumerate(zip(df_wce['model'], df_wce['accuracy'])):
        ax1.text(acc + 0.5, i, f'{acc:.2f}%', va='center', fontweight='bold')
    
    # Focal Loss
    df_focal = df[df['loss'] == 'Focal Loss'].sort_values('accuracy', ascending=False)
    ax2.barh(df_focal['model'], df_focal['accuracy'], color='lightcoral')
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Focal Loss', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (model, acc) in enumerate(zip(df_focal['model'], df_focal['accuracy'])):
        ax2.text(acc + 0.5, i, f'{acc:.2f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison_by_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print_colored(f"\n✓ Comparison plots saved to {save_dir}/", Colors.BRIGHT_GREEN)


def print_summary_table(df):
    """Print summary table of results."""
    print_header("EXPERIMENT RESULTS SUMMARY")
    
    # Sort by accuracy
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    print(f"\n{'Rank':<6} {'Model':<18} {'Loss':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("="*85)
    
    for i, row in enumerate(df_sorted.itertuples(), 1):
        color = Colors.BRIGHT_GREEN if i == 1 else Colors.WHITE
        print_colored(
            f"{i:<6} {row.model:<18} {row.loss:<15} {row.accuracy:<8.2f} {row.precision:<8.2f} {row.recall:<8.2f} {row.f1:<8.2f}",
            color
        )
    
    print("="*85)
    
    # Best model
    best = df_sorted.iloc[0]
    print_colored(f"\n🏆 BEST MODEL: {best['model']} with {best['loss']}", Colors.BRIGHT_GREEN, bold=True)
    print_colored(f"   Accuracy: {best['accuracy']:.2f}%", Colors.BRIGHT_GREEN)


def save_summary_csv(df, save_dir='results'):
    """Save summary to CSV."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = save_dir / 'summary.csv'
    df.to_csv(csv_path, index=False)
    print_colored(f"✓ Summary saved to {csv_path}", Colors.BRIGHT_BLUE)


def main():
    print_header("EXPERIMENT RESULTS COMPARISON")
    
    # Load results
    print_colored("Loading experiment results...", Colors.BRIGHT_CYAN)
    df = load_experiment_results()
    
    if len(df) == 0:
        print_colored("ERROR: No experiment results found!", Colors.BRIGHT_RED)
        print_colored("Run experiments first using: ./scripts/experiments/run_all.sh", Colors.BRIGHT_YELLOW)
        return
    
    print_colored(f"✓ Loaded {len(df)} experiments\n", Colors.BRIGHT_GREEN)
    
    # Print summary table
    print_summary_table(df)
    
    # Save CSV
    save_summary_csv(df)
    
    # Plot comparisons
    print_colored("\nGenerating comparison plots...", Colors.BRIGHT_CYAN)
    plot_comparison(df)
    
    # Final message
    print_header("COMPARISON COMPLETED")
    print_colored("Results saved to: results/", Colors.BRIGHT_GREEN, bold=True)
    print_colored("  - summary.csv", Colors.BRIGHT_BLUE)
    print_colored("  - accuracy_comparison.png", Colors.BRIGHT_BLUE)
    print_colored("  - metrics_comparison.png", Colors.BRIGHT_BLUE)
    print_colored("  - model_comparison_by_loss.png\n", Colors.BRIGHT_BLUE)


if __name__ == "__main__":
    main()
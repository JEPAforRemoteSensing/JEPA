import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the 19 BigEarthNet classes
CLASSES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]

# Shortened class names for better visualization
SHORT_CLASSES = [
    "Agro-forestry",
    "Arable land",
    "Beaches/dunes",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation",
    "Coniferous forest",
    "Industrial/commercial",
    "Inland waters",
    "Inland wetlands",
    "Agriculture w/ vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors/heathland",
    "Natural grassland",
    "Pastures",
    "Permanent crops",
    "Transitional woodland",
    "Urban fabric",
]

def main():
    # Load the Serbia metadata
    df = pd.read_parquet('data/BEN_14K/serbia_metadata.parquet')
    
    # Convert one_hot_labels to numpy array and sum across all samples
    one_hot_matrix = np.array(df['one_hot_labels'].tolist())
    class_counts = one_hot_matrix.sum(axis=0)
    
    # Create DataFrame for easier manipulation
    class_df = pd.DataFrame({
        'class': SHORT_CLASSES,
        'full_name': CLASSES,
        'count': class_counts
    }).sort_values('count', ascending=True)
    
    # Calculate percentages
    total_samples = len(df)
    class_df['percentage'] = (class_df['count'] / total_samples * 100).round(2)
    
    # Print statistics
    print("=" * 60)
    print(f"Class Distribution in Serbia Metadata ({total_samples} samples)")
    print("=" * 60)
    for _, row in class_df.iterrows():
        print(f"{row['class']:25s}: {row['count']:6d} ({row['percentage']:5.2f}%)")
    print("=" * 60)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_df)))
    bars = ax.barh(class_df['class'], class_df['count'], color=colors)
    
    # Add count labels on bars
    for bar, count, pct in zip(bars, class_df['count'], class_df['percentage']):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{int(count):,} ({pct:.1f}%)', va='center', fontsize=9)
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Land Cover Class', fontsize=12)
    ax.set_title(f'Class Imbalance in Serbia BigEarthNet Subset\n(Total: {total_samples:,} samples)', 
                 fontsize=14, fontweight='bold')
    
    # Add grid for readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = 'data/BEN_14K/serbia_class_imbalance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()

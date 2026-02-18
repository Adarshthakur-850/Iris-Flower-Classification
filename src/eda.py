import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

def perform_eda(df):
    print("Performing EDA...")
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # Pair Plot
    plt.figure(figsize=(10, 8))
    sns.pairplot(df, hue='species_name')
    plt.savefig("plots/pairplot.png")
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    # Drop non-numeric for correlation
    numeric_df = df.drop(columns=['species_name'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation")
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()
    
    # Feature Distributions
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(df.columns[:-2]): # Exclude species and species_name
        plt.subplot(2, 2, i+1)
        sns.histplot(data=df, x=col, hue='species_name', kde=True)
        plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig("plots/feature_distributions.png")
    plt.close()

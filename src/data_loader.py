import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    print("Loading Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    # Map target integers to species names for readability during EDA
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print(f"Dataset shape: {df.shape}")
    return df

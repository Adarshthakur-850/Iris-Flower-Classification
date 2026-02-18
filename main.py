import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import perform_eda
from src.model_trainer import train_models
from src.evaluation import evaluate_model
from src.visualization import plot_decision_boundaries, visualize_results

# For visualization, we need fresh models to train on 2D data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def main():
    print("Starting Iris classification pipeline...")
    
    # 1. Load Data
    df = load_data()
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Preprocessing
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # 4. Model Training
    models, best_model, best_model_name = train_models(X_train, y_train)
    
    # 5. Evaluation
    metrics_list = []
    for name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        metrics_list.append(metrics)
        
    metrics_df = pd.DataFrame(metrics_list)
    print("\nModel Leaderboard:")
    print(metrics_df)
    metrics_df.to_csv("models/metrics.csv", index=False)
    
    # 6. Visualization
    # Scatter of actual vs predicted for Best Model
    best_y_pred = best_model.predict(X_test)
    visualize_results(X_test, y_test, best_y_pred)
    
    # Decision Boundaries (Using first two features: Sepal Length, Sepal Width)
    # We need to retrain on just these 2 features for 2D plotting
    print("Generating decision boundary plots...")
    X_2d = df.iloc[:, :2].values
    y_2d = df['species'].values
    
    # Feature names for axes
    f_names_2d = feature_names[:2]
    
    viz_models = {
        'LogisticRegression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'DecisionTree': DecisionTreeClassifier()
    }
    
    for name, model in viz_models.items():
        try:
            plot_decision_boundaries(X_2d, y_2d, model, f_names_2d, f"decision_boundary_{name}.png")
        except Exception as e:
            print(f"Could not plot boundary for {name}: {e}")

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pipeline Failed: {e}")

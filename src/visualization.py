import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.inspection import DecisionBoundaryDisplay

def plot_decision_boundaries(X, y, model, feature_names, filename):
    # X must be 2D for this
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # Fit model on 2D data (only for visualization purposes)
    model.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict",
        cmap=plt.cm.RdYlBu,
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.6,
        ax=ax
    )
    
    # Plot training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.RdYlBu)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f"Decision Boundary: {type(model).__name__}")
    plt.savefig(f"plots/{filename}")
    plt.close()

def visualize_results(X_test, y_test, y_pred):
    # Scatter plot of actual vs predicted (using first two features)
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', label='Actual', alpha=0.6)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', label='Predicted', alpha=0.6)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Actual vs Predicted (Test Set)")
    plt.legend()
    plt.savefig("plots/prediction_scatter.png")
    plt.close()

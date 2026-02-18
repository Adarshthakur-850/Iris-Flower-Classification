from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os
import numpy as np

def train_models(X_train, y_train):
    print("Training models...")
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'DecisionTree': DecisionTreeClassifier()
    }
    
    results = {}
    best_score = -1
    best_model_name = ""
    best_model = None
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = mean_cv_score
        print(f"{name} CV Accuracy: {mean_cv_score:.4f}")
        
        # Fit on full train set
        model.fit(X_train, y_train)
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model_name = name
            best_model = model
            
    print(f"Best Model: {best_model_name} with CV Accuracy: {best_score:.4f}")
    joblib.dump(best_model, f"models/best_model.pkl")
    
    return models, best_model, best_model_name

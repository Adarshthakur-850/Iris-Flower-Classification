import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    print("Preprocessing data...")
    
    if df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        df.dropna(inplace=True)
    
    X = df.drop(columns=['species', 'species_name'])
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

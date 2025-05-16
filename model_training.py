import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from data_collection import fetch_stock_data

def analyze_prediction(model, feature_values, feature_names, prediction, probabilities):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
        'value': feature_values[0]
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(3)
    
    analysis = []
    for _, feature in top_features.iterrows():
        if feature['importance'] > 0.1:
            analysis.append(f"{feature['feature'].upper()}: {feature['value']:.2f} (Importance: {feature['importance']:.2%})")
    
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    if confidence < 0.6:
        strength = "UNCERTAIN"
    elif confidence < 0.75:
        strength = "MODERATE"
    else:
        strength = "STRONG"
    
    return {
        'prediction': 'UP' if prediction == 1 else 'DOWN',
        'confidence': confidence,
        'strength': strength,
        'key_factors': analysis,
        'raw_probabilities': probabilities
    }

def train_model(ticker):
    data = fetch_stock_data(ticker)
    
    feature_columns = ['rsi', 'macd', 'macd_signal', 'macd_hist', 
                      'upper_band', 'middle_band', 'lower_band',
                      'volume_sma', 'obv', 'adx', 'cci']
    
    X = data[feature_columns]
    y = data['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [20, 30, 40],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='f1', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_scaled)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n📈 Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"\nBest Parameters: {grid_search.best_params_}")
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return best_model, scaler, feature_columns

def predict_next_day(model, scaler, ticker, feature_columns):
    data = fetch_stock_data(ticker, period="1d", interval="1d")
    
    X = data[feature_columns]
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[-1]
    probability = model.predict_proba(X_scaled)[-1]
    
    analysis = analyze_prediction(model, X.values, feature_columns, prediction, probability)
    
    print("\n🔍 Prediction Analysis:")
    print("=" * 50)
    print(f"Prediction: {analysis['prediction']} ({analysis['strength']})")
    print(f"Confidence: {analysis['confidence']:.2%}")
    print("\nKey Factors Influencing Prediction:")
    for factor in analysis['key_factors']:
        print(f"- {factor}")
    print("\nRaw Probabilities:")
    print(f"UP: {analysis['raw_probabilities'][1]:.2%}")
    print(f"DOWN: {analysis['raw_probabilities'][0]:.2%}")
    print("=" * 50)
    
    return analysis

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(file_path):
    """Load processed data."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded processed data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def train_model():
    """Train and evaluate models with hyperparameter tuning."""
    # Load processed data
    data = load_processed_data("data/processed/processed_data.csv")
    X = data.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = data['is_high_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models and parameter grids
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(),
            'params': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(),
            'params': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
        }
    }
    
    best_model = None
    best_roc_auc = 0
    best_model_name = ""
    
    # Start MLflow run
    with mlflow.start_run():
        for name, config in models.items():
            logger.info(f"Training {name}...")
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='roc_auc', 
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test)
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            
            # Log metrics and model to MLflow
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(grid_search.best_estimator_, name)
            
            logger.info(f"{name} metrics: {metrics}")
            
            # Update best model
            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_model = grid_search.best_estimator_
                best_model_name = name
        
        # Register best model
        mlflow.sklearn.log_model(best_model, f"best_model_{best_model_name}")
        logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_roc_auc}")
    
    return best_model

if __name__ == "__main__":
    mlflow.set_experiment("credit_risk_model")
    train_model()
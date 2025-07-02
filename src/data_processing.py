import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from xverse.transformer import WOETransformer
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load raw data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def extract_temporal_features(df):
    """Extract temporal features from TransactionStartTime."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    logger.info("Extracted temporal features")
    return df

def create_aggregate_features(df):
    """Create RFM and other aggregate features per CustomerId."""
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'TransactionStartTime': ['min', 'max']
    }).reset_index()
    
    agg_features.columns = [
        'CustomerId', 
        'TotalTransactionAmount', 
        'AvgTransactionAmount', 
        'TransactionCount', 
        'StdTransactionAmount',
        'FirstTransactionTime', 
        'LastTransactionTime'
    ]
    
    # Calculate Recency (days since last transaction)
    snapshot_date = datetime(2025, 6, 30)
    agg_features['Recency'] = (snapshot_date - agg_features['LastTransactionTime']).dt.days
    
    # Fill NaN for StdTransactionAmount
    agg_features['StdTransactionAmount'] = agg_features['StdTransactionAmount'].fillna(0)
    
    logger.info("Created aggregate features")
    return agg_features

def cluster_rfm_features(agg_df):
    """Cluster customers based on RFM features using K-Means."""
    rfm_features = ['Recency', 'TransactionCount', 'TotalTransactionAmount']
    rfm_data = agg_df[rfm_features]
    
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    agg_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (low frequency, low monetary, high recency)
    cluster_summary = agg_df.groupby('Cluster')[rfm_features].mean()
    high_risk_cluster = cluster_summary[
        (cluster_summary['TransactionCount'] == cluster_summary['TransactionCount'].min()) &
        (cluster_summary['TotalTransactionAmount'] == cluster_summary['TotalTransactionAmount'].min())
    ].index[0]
    
    agg_df['is_high_risk'] = (agg_df['Cluster'] == high_risk_cluster).astype(int)
    logger.info(f"Identified high-risk cluster: {high_risk_cluster}")
    return agg_df[['CustomerId', 'is_high_risk']]

def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Build a preprocessing pipeline for numerical and categorical features."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('woe', WOETransformer()),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def process_data(input_path, output_path):
    """Main function to process raw data and save processed data with target variable."""
    # Load data
    df = load_data(input_path)
    
    # Extract temporal features
    df = extract_temporal_features(df)
    
    # Create aggregate features
    agg_df = create_aggregate_features(df)
    
    # Cluster RFM features to create proxy target
    target_df = cluster_rfm_features(agg_df)
    
    # Merge with original data
    processed_df = df.merge(agg_df, on='CustomerId', how='left').merge(target_df, on='CustomerId', how='left')
    
    # Define feature columns
    numerical_cols = [
        'TotalTransactionAmount', 'AvgTransactionAmount', 'TransactionCount',
        'StdTransactionAmount', 'Recency', 'TransactionHour', 'TransactionDay', 'TransactionMonth'
    ]
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    
    # Build and apply preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    processed_data = preprocessor.fit_transform(processed_df[numerical_cols + categorical_cols])
    
    # Create output DataFrame
    feature_names = (
        numerical_cols + 
        list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols))
    )
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    processed_df['CustomerId'] = df['CustomerId'].reset_index(drop=True)
    processed_df['is_high_risk'] = target_df['is_high_risk'].reset_index(drop=True)
    
    # Save processed data
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data with target variable to {output_path}")
    
    return processed_df, preprocessor

if __name__ == "__main__":
    input_path = "data/raw/xente_data.csv"
    output_path = "data/processed/processed_data.csv"
    processed_df, preprocessor = process_data(input_path, output_path)
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ... (Your load_data and preprocess_data functions remain the same as corrected previously for sparse_output) ...

def load_data(raw_path):
    """Load raw data from the specified path."""
    return pd.read_csv(raw_path)

def preprocess_data(df): # This df is the one that comes from load_data
    """Preprocess data and return a pipeline."""
    # Aggregate features by CustomerId
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'Value': ['sum', 'mean']
    }).reset_index()
    agg_features.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount', 'TotalValue', 'AvgValue']
    df = df.merge(agg_features, on='CustomerId', how='left')

    # Extract time features from TransactionStartTime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year

    # Define numerical and categorical columns
    # These lists assume the columns are now present in 'df' after the above steps
    numeric_features = [
        'Amount', 'Value', 'TotalAmount', 'AvgAmount', 'TransactionCount',
        'StdAmount', 'TotalValue', 'AvgValue', 'TransactionHour',
        'TransactionDay', 'TransactionMonth', 'TransactionYear'
    ]
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']

    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns not explicitly transformed (like CustomerId, TransactionId, TransactionStartTime if not in feature lists)
    )

    # Return the preprocessor. The actual fit_transform happens outside this function.
    return preprocessor

if __name__ == "__main__":
    # Example usage
    raw_data_path = 'data/raw/data.csv'
    df = load_data(raw_data_path)

    # NEW: Perform the feature engineering steps on the DataFrame *before* passing it to the pipeline
    # The preprocess_data function as currently written already does this and returns the preprocessor.
    # What's critical is that the *df* that the preprocessor is trained on and transforms
    # contains all the columns that the preprocessor expects.

    # Option 1: Integrate the feature engineering *into* the data loading/preparation before pipeline
    # The `preprocess_data` function is currently structured to do the feature engineering
    # AND return the ColumnTransformer. We need to split this for the pipeline to work correctly.

    # Let's refactor `preprocess_data` to return the modified df AND the preprocessor.
    # Or, simpler: perform the feature engineering directly on df here, then create the pipeline.

    # --- Refactored approach: ---

    # 1. Load data
    raw_df = load_data(raw_data_path) # Use a different name to distinguish from modified df

    # 2. Perform feature engineering directly on the raw_df to create engineered features
    #    This part is essentially what was implicitly happening at the start of preprocess_data
    #    but needs to be applied to the DataFrame BEFORE it enters the ColumnTransformer.

    # Aggregate features by CustomerId
    agg_features = raw_df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'Value': ['sum', 'mean']
    }).reset_index()
    agg_features.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount', 'TotalValue', 'AvgValue']
    df_engineered = raw_df.merge(agg_features, on='CustomerId', how='left')

    # Extract time features from TransactionStartTime
    df_engineered['TransactionStartTime'] = pd.to_datetime(df_engineered['TransactionStartTime'])
    df_engineered['TransactionHour'] = df_engineered['TransactionStartTime'].dt.hour
    df_engineered['TransactionDay'] = df_engineered['TransactionStartTime'].dt.day
    df_engineered['TransactionMonth'] = df_engineered['TransactionStartTime'].dt.month
    df_engineered['TransactionYear'] = df_engineered['TransactionStartTime'].dt.year

    # 3. Define numerical and categorical columns for the *ColumnTransformer*
    #    These lists now refer to columns that *exist* in df_engineered
    numeric_features = [
        'Amount', 'Value', 'TotalAmount', 'AvgAmount', 'TransactionCount',
        'StdAmount', 'TotalValue', 'AvgValue', 'TransactionHour',
        'TransactionDay', 'TransactionMonth', 'TransactionYear'
    ]
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']

    # 4. Create preprocessing steps (pipeline definition)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns not explicitly transformed
    )

    # 5. Fit and transform the engineered DataFrame
    #    The preprocessor (pipeline) now receives a DataFrame that contains all expected columns
    processed_data = preprocessor.fit_transform(df_engineered)

    # 6. Reconstruct the DataFrame with correct column names
    #    It's crucial to get the feature names out from the fitted preprocessor
    processed_feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed_data, columns=processed_feature_names)

    # 7. Concatenate CustomerId and TransactionId back
    #    Make sure the indices align correctly. Resetting index is good practice.
    #    Also, ensure these columns were passed through by ColumnTransformer ('remainder'='passthrough')
    #    and select them from the original (or engineered) DataFrame.
    
    # We need to explicitly get the non-transformed columns from df_engineered
    # For remainder='passthrough', columns that are not explicitly selected by a transformer
    # will be passed through to the output, appearing at the end of the transformed array.
    # So, we should select 'CustomerId' and 'TransactionId' from the original df_engineered
    # to combine them, ensuring their index aligns.
    
    # Identify columns that should have been passed through
    # These would be the columns in df_engineered that are NOT in numeric_features or categorical_features
    passthrough_cols_df = df_engineered.drop(columns=numeric_features + categorical_features + ['TransactionStartTime'], errors='ignore')
    
    # Reset index for safe concatenation
    processed_df = pd.DataFrame(processed_data, columns=processed_feature_names)
    df_ids_to_concat = passthrough_cols_df[['CustomerId', 'TransactionId']].reset_index(drop=True) # Assuming these are the ones
    
    # Now concatenate:
    # Ensure processed_df also has its index reset if it's not already 0-based sequential
    processed_df = pd.concat([df_ids_to_concat, processed_df.reset_index(drop=True)], axis=1)

    processed_df.to_csv('data/processed/processed_data.csv', index=False)
    print("Processed data saved to data/processed/processed_data.csv")
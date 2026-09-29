import pandas as pd 

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )
    
    mask = (
        df['TotalCharges'].isna()
        & (df['tenure'] == 0)
    )
    
    df.loc[mask, 'TotalCharges'] = 0
    
    return df
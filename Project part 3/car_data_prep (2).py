import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

def prepare_data(df, fit=False, preprocessor=None):
    df['Hand'] = pd.to_numeric(df['Hand'], errors='coerce').fillna(1).astype(int)
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce').fillna(1673).astype(float)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    columns_to_drop = ['Prev_ownership', 'Curr_ownership', 'Color', 'Test', 'Supply_score', 'City', 'Area']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])     # Drop columns if they exist in the DataFrame
    
    model_mapping = {
        'אונסיס': 'אוונסיס',
        'sx4 קרוסאובר': 'קרוסאובר',
        'sx4': 'קרוסאובר',
        'קאונטרימן': 'קאנטרימן',
        'פלואנס חשמלי': 'פלואנס',
        'קליאו אסטייט': 'קליאו',
        'קליאו דור 4': 'קליאו',
        'מגאן אסטייט / גראנד טור': 'מגאן',
        "גראנד, וויאג'ר": "וויאג'ר",
        'סיטיגו / citygo': 'סיטיגו',
        'סיוויק סדאן החדשה': 'סיוויק',
        "סיוויק האצ'בק החדשה": 'סיוויק',
        "סיוויק האצ'בק": 'סיוויק',
        'סיוויק סדאן': 'סיוויק',
        'civic': 'סיוויק',
        'accord': 'אקורד',
        'jazz': "ג'אז",
        "ג'אז הייבריד": "ג'אז",
        'insight': 'אינסייט',
        'קורסה החדשה': 'קורסה',
        'קרוז החדשה': 'קרוז',
        'ראפיד ספייסבק': 'ראפיד',
        'אוקטביה קומבי': 'אוקטביה',
        'אוקטביה ספייס': 'אוקטביה',
        'סיטיגו / citygo': 'סיטיגו',
        "ג'וק juke": "ג'וק",
        'פאביה ספייס': 'פאביה',
        'סוויפט החדשה': 'סוויפט',
        'e-class קופה / קבריולט': 'e-class',
        'i30cw':'i30',
        'נירו ev': 'נירו',
        'נירו phev': 'נירו',
        'לנסר הדור החדש': 'לנסר',
        'לנסר ספורטבק': 'לנסר',
        'c-class taxi': 'c-class',
        'c-class קופה': 'c-class',
        'e- class': 'e-class',
        'מוקה x': 'מוקה',
        'פאסאט cc': 'פאסאט',
        'גולף פלוס': 'גולף',
        'גולף gti': 'גולף',
        'חיפושית חדשה': 'חיפושית',
        'מיטו / mito': 'מיטו',
    }

    def is_republicated(df):
        df['Is_republicated'] = (df['Cre_date'] == df['Repub_date']).astype(int)
        df.drop(columns=['Cre_date', 'Repub_date'], inplace=True)
        return df

    def model_cleaner(df):
        df['manufactor'] = df['manufactor'].str.replace('Lexsus', 'לקסוס')
        df['model'] = df['model'].str.strip()
        df['model'] = df['model'].str.replace(r'\s+', ' ', regex=True)
        df['model'] = df['model'].str.replace(r'\r\n', '', regex=True)
        df['model'] = df['model'].str.replace('`', "'")
        df['model'] = df['model'].str.lower()
        unique_manufactors = df['manufactor'].unique().tolist()
        for brand in unique_manufactors:
            df['model'] = df['model'].str.replace(brand, '', regex=False)
        df['model'] = df['model'].str.strip()
        df['model'] = df['model'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        df['model'] = df['model'].replace(model_mapping)
        return df

    def is_luxury(df):
        luxury_brands = ['אאודי', 'מרצדס', 'לקסוס', 'וולוו', 'ב.מ.וו','לקסוס','יגואר','ביואיק','פורשה','אינפיניטי','קאדילק','סמארט','מזארטי','פרארי','סאנגיונג','אוטוביאנקי','טסלה']
        df['Is_luxury'] = df['manufactor'].isin(luxury_brands).astype(int)
        df.drop(columns=['manufactor'], inplace=True)
        return df

    def fix_mileage_year(df):
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Km'] = pd.to_numeric(df['Km'], errors='coerce').fillna(0).astype(int)
        current_year = 2024
        mean_km_per_year = 15000
        df['Km'] = df.apply(lambda row: row['Km'] * 1000 if len(str(row['Km'])) <= 3 and row['Year'] >= current_year - 1 else row['Km'], axis=1)
        df['Age'] = current_year - df['Year']
        df['Km'] = df.apply(lambda row: row['Km'] if pd.notnull(row['Km']) and row['Km'] != 0 else row['Age'] * mean_km_per_year, axis=1)
        df.drop(columns=['Year'], inplace=True)
        return df

    def gear_engine_encoder(df):
        df['Engine_type'] = df['Engine_type'].fillna('בנזין').replace({'טורבו דיזל': 'דיזל', 'היבריד': 'היברידי'})
        df['Is_Diesel'] = df['Engine_type'].isin(['דיזל', 'טורבו דיזל']).astype(int)
        df['Is_Hybrid_Electric'] = df['Engine_type'].isin(['היברידי', 'חשמלי']).astype(int)
        df['Gear'] = df['Gear'].fillna('אוטומט').replace(['לא מוגדר', 'אוטומטית'], 'אוטומט')
        df['Is_Automatic'] = df['Gear'].isin(['טיפטרוניק', 'אוטומט']).astype(int)
        return df

    def capacity_engine_cleaner(df):
        df['capacity_Engine'] = df['capacity_Engine'].astype(str).str.replace(',', '').astype(float)
        df['capacity_Engine'] = df['capacity_Engine'].fillna(1673)
        df.loc[df['Engine_type'] == 'גז', 'capacity_Engine'] = 1500
        df.drop(columns=['Gear', 'Engine_type'], inplace=True)
        return df

    def outliers_adjuster(df):
        df['Age'] = df['Age'].clip(upper=20)
        df['Hand'] = df['Hand'].clip(upper=7)
        df['capacity_Engine'] = df['capacity_Engine'].clip(lower=800, upper=8000)
        df['Km'] = df['Km'].clip(upper=400000)
        return df
        
    def outliers_cleaner(df):
        df = df[df['Age'] <= 20]
        df = df[df['Hand'] <= 7]
        df = df[df['capacity_Engine'].between(800, 8000)]
        df = df[df['Km'] <= 400000]
        df = df[df['Pic_num'] <= 25]
        return df

    
    # Start of data preparation
    df = df.copy()
    binary_features = ['Is_republicated', 'Is_luxury', 'Is_Automatic', 'Is_Hybrid_Electric']
    numeric_features = ['Hand', 'Km', 'Age', 'capacity_Engine']
    categorical_features = ['model']
    target = ['Price']
    
    if fit:
    # Fit the preprocessor if fitting is required
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('scaler', RobustScaler())
                ]), numeric_features),
                ('bin', SimpleImputer(strategy='constant', fill_value=0), binary_features),
                ('cat', Pipeline([
                    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
                ]), categorical_features),
            ]
        )
        
        t_scaler = RobustScaler()
        
        df = df.drop_duplicates(subset=['model', 'Year', 'Gear', 'Cre_date', 'Hand', 'Description'], keep='first')
        df = is_republicated(df)
        df = model_cleaner(df)
        df = is_luxury(df)
        df = fix_mileage_year(df)
        df = gear_engine_encoder(df)
        df = capacity_engine_cleaner(df)
        df = outliers_cleaner(df)
        processed_data = pd.DataFrame(preprocessor.fit_transform(df.drop(columns = 'Price')))    
        processed_data['Price'] = t_scaler.fit_transform(df.Price.values.reshape(-1, 1))
                                      
        numeric_columns = numeric_features + binary_features   # Get feature names after preprocessing
        cat_columns = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
        processed_data.columns = numeric_columns + list(cat_columns) + target # Include the target only for training process
        return processed_data, preprocessor, t_scaler

    else:
        # Ensure the preprocessor is already fitted
        df = is_republicated(df)
        df = model_cleaner(df)
        df = is_luxury(df)
        df = fix_mileage_year(df)
        df = gear_engine_encoder(df)
        df = capacity_engine_cleaner(df)
        df = outliers_adjuster(df)
        processed_data = pd.DataFrame(preprocessor.transform(df))
        
        numeric_columns = numeric_features + binary_features   # Get feature names after preprocessing
        cat_columns = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
        processed_data.columns = numeric_columns + list(cat_columns)
        return processed_data
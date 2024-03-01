from timeit import default_timer as timer
import time
import numpy as np
import pandas as pd
from math import isnan
from sklearn import preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class PreProcessor:
   
    def __init__(self):
        self.unique_count = 1
        self.start_time=time.time()
    
    def RemoveIrrelevantColumn(self,df):
        #regex expressions for various formats of dates
        start_time=time.time()
        regex_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
        r'\b\d{4}/\d{2}/\d{2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b',
        r'\b\d{10,13}\b',
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
        ]
        count=0
        for column in df.columns:
            if df[column].nunique(dropna=True)==self.unique_count:
                print(f'Column :{column} is removed')
                count=count+1
                df=df.drop(column,axis=1)
        print(f'{count} irrelavant columns found.')
        columns_to_remove = []
    
        for column in df.columns:
            # Check if any value in the column matches any regex pattern
            if any(df[column].astype(str).str.match(pattern).any() for pattern in regex_patterns):
                columns_to_remove.append(column)
        df = df.drop(columns=columns_to_remove)
        end_time=time.time()
        time_elp= end_time-start_time
        print(f'Execution time for RemoveIrrelevantColumn is {time_elp} seconds')
        
        return df

        

    def HandlingMissingData(self,df,num_strategy='most_frequent',cat_strategy='knn',n_neighbors=3,null_threshold=0.1):
        start_time=time.time()
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = [column for column in df.columns if column not in num_cols]
        
        for column in num_cols:
            null_val=df[column].isnull().mean()
            if(null_val!=0 and null_val<=null_threshold):
                print(f'{null_val}% NaN values found on column: {column}')
                df=df.dropna(subset=[column])
                df= df.reset_index(drop=True)

        imputer=None
        if(num_strategy=='knn'):
            imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            if num_strategy in ['mean','median','mode','most_frequent']:
                imputer = SimpleImputer(strategy=num_strategy)
            else:
                print('Invalid imputer strategy specified :{}\nDefault strategy Mean is applied',num_strategy)
                imputer = SimpleImputer(strategy='mean')
        print('imputation process started...')
        for feature in num_cols:
            if df[feature].isna().sum().sum() != 0:
                try:
                    df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))
                    if (df[feature].fillna(-9999) % 1  == 0).all():
                        df[feature] = df_imputed
                        # round back to INTs, if original data were INTs
                        df[feature] = df[feature].round()
                        df[feature] = df[feature].astype('Int64')                                        
                    else:
                        df[feature] = df_imputed
                except:
                    print('imputation failed for feature "{}"',feature)
        if(cat_strategy=='knn'):
            imputer = KNNImputer(n_neighbors=n_neighbors)
        elif(cat_strategy=='logreq'):
            df = PreProcessor.LogisticRegressionImputer(
                columns=cat_cols,
                df=df
            )
            return df
        else:
            imputer = SimpleImputer(strategy='most_frequent')
        
        for feature in cat_cols:
            if df[feature].isna().sum()!= 0:
                try:
                    mapping = dict()
                    mappings = {k: i for i, k in enumerate(df[feature].dropna().unique(), 0)}
                    mapping[feature] = mappings
                    df[feature] = df[feature].map(mapping[feature])

                    df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)), columns=[feature])    

                    # round to integers before mapping back to original values
                    df[feature] = df_imputed
                    df[feature] = df[feature].round()
                    df[feature] = df[feature].astype('Int64')  

                    # map values back to original
                    mappings_inv = {v: k for k, v in mapping[feature].items()}
                    df[feature] = df[feature].map(mappings_inv)
                except:
                    print('Imputation failed for feature "{}"',  feature)
        end_time=time.time()    
        time_elp= end_time-start_time
        print(f'Execution time for HandlingMissingData is {time_elp} seconds')
        return df
    
    
    def normalization(self,df):
        sc = StandardScaler()
        normalize_columns = []
        for column in df.columns:
            if (df[column].dtype == 'int64' or df[column].dtype == 'float64') and df[column].nunique() > 10:
                normalize_columns.append(column)
        df[normalize_columns] = sc.fit_transform(df[normalize_columns])
        # Is normalization Done well
        normalized_feature = df[normalize_columns]
        mean_normalized = np.mean(normalized_feature)
        std_dev_normalized = np.std(normalized_feature)
        print("Mean of normalized feature:", mean_normalized)
        print("Standard deviation of normalized feature:", std_dev_normalized)
        return df
    
    
    
    
    def encoding(self,df):   
        start_time=time.time()
        lable_encoder = preprocessing.LabelEncoder()
        object_columns = []
        for column in df.columns:
                if df[column].dtype == 'object' and df[column].nunique()<3: #binary 
                    df[column]=lable_encoder.fit_transform(df[column])
                    object_columns.append(column)
                elif df[column].dtype == 'object' and (df[column].nunique()>2): #Multi-class 
                    df[column]=lable_encoder.fit_transform(df[column])
                    object_columns.append(column)
                elif df[column].dtype == 'bool':
                    df[column]=lable_encoder.fit_transform(df[column])
                    object_columns.append(column)
                else:
                    pass
        
        end_time=time.time()
        time_elp= end_time-start_time
        print("Encoding process completed! It returns df and Object-col-list")
        print(f'Execution time for encoding is {time_elp} seconds') 
        total_time_elp= end_time-self.start_time
        print(f'Total Execution time for Preprocessing is {total_time_elp} seconds')     
        
        return df,object_columns,total_time_elp
    
    
    
        
    def LogisticRegressionImputer(self,columns,df):
             for feature in columns:
                 try:
                     test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                     train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                     if len(test_df.index) != 0:
                         pipe = make_pipeline(StandardScaler(), LogisticRegression())  
                         y = train_df[feature]
                         train_df.drop(feature, axis=1, inplace=True)
                         test_df.drop(feature, axis=1, inplace=True)   
                         model = pipe.fit(train_df, y)
                       
                         pred = model.predict(test_df) # predict values
                         test_df[feature]= pred
                         if (df[feature].fillna(-9999) % 1  == 0).all():
                             # round back to INTs, if original data were INTs
                             test_df[feature] = test_df[feature].round()
                             test_df[feature] = test_df[feature].astype('Int64')
                             df[feature].update(test_df[feature])                             
                         print('LOGREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                 except:
                     print('LOGREG imputation failed for feature "{}"', feature)
                 for feature in df.columns: 
                     try:
                         # map categorical feature values back to original
                         mappings_inv = {v: k for k, v in mapping[feature].items()}
                         df[feature] = df[feature].map(mappings_inv)
                     except:
                         pass     
             return df
    
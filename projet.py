import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('data.csv')
#print(data.head(10))
#print(data.columns)

data.drop(['Div', 'Date', 'Time'], axis=1, inplace=True) # On retire ces colonnes car inutiles

#Detection des valeurs manquantes 
missing_value = data.isnull().sum()
#print("nb value manquantes :  " + str(missing_value)) # Pas de valeurs manquantes dans ce dataset

match_stats_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'] #Colonnes à analyser pour détecter les valeurs aberrantes
outliers_info = {}

for column in match_stats_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1 # Calcul de la dispersion des données
    lower_bound = Q1 - 1.5 * IQR # Définition des bornes pour la prise en compte des données 
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)] # On stocke les outliers dans un tableau 

    # Désormais, on cappe les valeurs pour atténuer l'impact des outliers, sans les supprimer pour autant

    data[column] = data[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    
    outliers_info[column] = { #On répertorie dans un dictionnaire les données analysées
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outliers Count': len(outliers),
        'Outliers Percentage': (len(outliers) / len(data)) * 100
    }

outliers_summary = pd.DataFrame(outliers_info).T
#print(outliers_summary)

# One-Hot Encoding pour les équipes et le résultat du match
data_encoded = pd.get_dummies(data, columns=['HomeTeam', 'AwayTeam', 'FTR'], drop_first=True)

#print(data.head(10))

numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

columns_to_exclude = ['HomeTeam', 'AwayTeam', 'FTR']  # Exemple de colonnes encodées à exclure
numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude]

# Initialiser le MinMaxScaler
scaler = MinMaxScaler()

# Appliquer la normalisation aux colonnes numériques pertinentes
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

print(data[numeric_columns].head())
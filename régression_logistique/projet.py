import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

############ TRAITEMENT DES DONNEES ##################
data = pd.read_csv('data.csv')

data.drop(['Div', 'Date', 'Time','HTR'], axis=1, inplace=True)

# On gère les valeurs aberrantes et hors quartile
match_stats_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
for column in match_stats_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

data_encoded = pd.get_dummies(data, columns=['HomeTeam', 'AwayTeam', 'FTR'], drop_first=True) 


#numeric_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist() # On normalise seulement les colonnes numériques de data et pas data_encoded pour éviter de normaliser les colonnes binaires
scaler = MinMaxScaler() 
data_encoded[numeric_columns] = scaler.fit_transform(data_encoded[numeric_columns])


data_encoded.replace([float('inf'), float('-inf')], pd.NA, inplace=True) # On supprime les valeurs infinies
data_encoded.fillna(data_encoded.mean(), inplace=True)

for column in data_encoded.columns:
    try:
        data_encoded[column] = data_encoded[column].astype(int)
    except ValueError as e:
        print(f"Erreur dans la colonne: {column} - {e}")


# Mettre à jour X et y
X = data_encoded.drop(['FTR_H','FTR_D'], axis=1)  # Caractéristiques
y = data_encoded['FTR_H']  # Nouvelle cible avec une seule colonnen --> on cible la victoire, nul ou défaite de l'équipe à domicile

########### SEPARATION DES DONNEES ET MODELE ##########
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrainement du modèle
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy du modèle :", accuracy)
print("Rapport de classification :\n", report)

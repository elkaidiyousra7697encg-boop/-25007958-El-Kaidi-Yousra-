# El Kaidi Yousra
<img src="2.jpg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

Numéro d'étudiant : 25007958

Classe : CAC2

#  COMPTE RENDU – PROJET D’ANALYSE DE LA CROISSANCE DU PIB

## 1. Contexte et Objectif du Projet

L’objectif de ce projet est d’analyser et de prédire la **croissance du Produit Intérieur Brut (PIB)** à partir de données macroéconomiques historiques issues d’un fichier CSV (*real-gdp-growth.csv*).

Le projet suit une démarche complète de **Data Science**, allant de la lecture et du nettoyage des données jusqu’à l’évaluation comparative de plusieurs modèles de régression.

---

## 2. Données Utilisées

### Description du jeu de données

Après résolution des problèmes de parsing du fichier CSV, le jeu de données final contient les variables suivantes :

* **Entity** : Nom du pays ou de l’entité
* **Code** : Code pays
* **Year** : Année
* **GDP Growth (Observations)** : Croissance réelle du PIB (variable cible)
* **GDP Growth (Forecasts)** : Prévisions de croissance du PIB (non utilisées pour la modélisation finale)

### Problèmes rencontrés

* Fichier CSV mal formaté (séparateurs et guillemets incohérents)
* Colonnes dupliquées et annotations inutiles
* Types de données incorrects (`object` au lieu de numérique)

---

## 3. Nettoyage et Préparation des Données

Les étapes suivantes ont été appliquées :

1. **Parsing robuste du CSV** avec gestion des lignes corrompues
2. **Suppression des colonnes d’annotations** non pertinentes
3. **Conversion des types** :

   * `Year` → entier
   * Croissance du PIB → numérique
4. **Gestion des valeurs manquantes** :

   * Suppression des lignes sans valeur cible
   * Remplacement des codes pays manquants par `Unknown`
5. **Encodage des variables catégorielles** (`Entity`, `Code`) via *One-Hot Encoding*
6. **Mise à l’échelle** de la variable `Year` avec `StandardScaler`

###  Code – Chargement et nettoyage des données

```python
import pandas as pd
import numpy as np
import csv

# Chargement robuste du fichier CSV
df = pd.read_csv(
    'real-gdp-growth.csv',
    sep=',',
    on_bad_lines='skip',
    header=None,
    quoting=csv.QUOTE_NONE
)

# Suppression de la ligne d'en-tête incluse dans les données
df = df.drop(index=0).reset_index(drop=True)

# Sélection des colonnes pertinentes
df = df[[0, 1, 2, 3, 5, 6]]
df.columns = [
    'Entity',
    'Code',
    'Year',
    'GDP_Growth_Observations',
    'GDP_Growth_Forecasts',
    'Forecast_Annotations'
]

# Conversion des types
df['Year'] = df['Year'].astype(int)
df['GDP_Growth_Observations'] = pd.to_numeric(df['GDP_Growth_Observations'], errors='coerce')

# Suppression des colonnes non pertinentes
df = df.drop(columns=['GDP_Growth_Forecasts', 'Forecast_Annotations'])
```

---

## 4. Analyse Exploratoire des Données (EDA)

### Constats principaux

* Les données couvrent plusieurs décennies et de nombreux pays
* La croissance du PIB présente une forte variabilité selon les périodes et les pays
* Les relations entre les variables explicatives et la croissance du PIB sont **non linéaires**

### Visualisations réalisées

* Histogrammes de la variable `Year`
* Distribution de la croissance du PIB
* Courbes temporelles pour plusieurs pays (États-Unis, Chine, Inde, Allemagne, Brésil)
* Matrice de corrélation pour les variables numériques

---

## 5. Modélisation

### Variable cible

* **y** : Croissance du PIB réel (observations)

### Variables explicatives

* Année (standardisée)
* Pays et code pays (encodage one-hot)

###  Code – Préparation des données pour la modélisation

```python
from sklearn.preprocessing import StandardScaler

# Suppression des lignes sans valeur cible
df_model = df.dropna(subset=['GDP_Growth_Observations']).copy()

# Séparation X / y
y = df_model['GDP_Growth_Observations']
X = df_model[['Entity', 'Code', 'Year']]

# Gestion des valeurs manquantes
X['Code'] = X['Code'].fillna('Unknown')

# Encodage one-hot
X = pd.get_dummies(X, columns=['Entity', 'Code'], drop_first=True)

# Standardisation de l'année
scaler = StandardScaler()
X['Year'] = scaler.fit_transform(X[['Year']])
```

---

## 6. Évaluation des Modèles

Les modèles ont été évalués à l’aide des métriques suivantes :

* **MSE** (Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **R²** (Coefficient de détermination)

###  Code – Entraînement et évaluation des modèles

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    })
```

| Modèle              | RMSE  | MAE  | R²    |
| ------------------- | ----- | ---- | ----- |
| Régression Linéaire | 42.67 | 3.44 | 0.03  |
| Ridge               | 42.63 | 3.43 | 0.03  |
| Lasso               | 44.17 | 3.54 | -0.00 |
| Arbre de Décision   | 43.46 | 3.56 | 0.02  |
| Random Forest       | 34.79 | 3.04 | 0.21  |
| Gradient Boosting   | 36.86 | 3.23 | 0.17  |

---

## 7. Analyse et Interprétation

* Les **modèles linéaires** (Linear, Ridge, Lasso) expliquent très peu la variance de la croissance du PIB
* Lasso souffre d’une régularisation trop forte, menant à un sous-ajustement
* L’**arbre de décision seul** est instable et peu performant
* Les **méthodes d’ensemble** (Random Forest et Gradient Boosting) capturent mieux les relations non linéaires

 **La Forêt Aléatoire est le meilleur modèle**, avec :

* L’erreur la plus faible
* Le meilleur pouvoir explicatif (R² = 0.21)

---

## 8. Conclusion Générale

Ce projet met en évidence l’importance :

* D’un **nettoyage rigoureux des données**
* D’une **analyse exploratoire approfondie**
* Du choix d’un **modèle adapté à la complexité des données**

Bien que la Forêt Aléatoire donne les meilleurs résultats, une large part de la variance reste inexpliquée. Cela suggère que la croissance du PIB dépend de nombreux facteurs supplémentaires (inflation, chômage, politiques économiques, chocs externes) non inclus dans ce jeu de données.

### Perspectives d’amélioration

* Optimisation des hyperparamètres (Grid Search)
* Ajout de nouvelles variables macroéconomiques
* Modèles plus avancés (XGBoost, LightGBM)

---

 **Conclusion finale :** Le projet démontre que les modèles d’ensemble sont plus efficaces pour prédire la croissance du PIB que les approches linéaires simples, tout en soulignant les limites inhérentes aux données disponibles.


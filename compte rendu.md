# El Kaidi Yousra
<img src="2.jpg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

Numéro d'étudiant : 25007958

Classe : CAC2

# COMPTE RENDU TECHNIQUE : PROJET DE MODÉLISATION DE LA CROISSANCE DU PIB

Ce rapport synthétise l'approche, la méthodologie et les résultats d'un projet de Machine Learning visant à prédire la croissance économique.

---

## 1. Contexte Métier et Objectif

### Le Problème (Business Case)
Comprendre et anticiper les fluctuations économiques est crucial pour l'élaboration de politiques budgétaires et les décisions d'investissement. L'objectif est d'expliquer et de prédire la dynamique du Produit Intérieur Brut (PIB).

### La Mission et la Variable Cible
* **Objectif :** Déterminer le meilleur modèle de régression pour prédire le pourcentage de variation annuel du PIB réel.
* **y (Target - Cible) :** `Gross domestic product, constant prices - Percent change - Observations` (Croissance du PIB à prix constants en pourcentage).

---

## 2. Le Laboratoire (Méthodologie du Code)

L'approche de modélisation a consisté à tester un ensemble de régresseurs pour identifier celui qui capture le mieux les motifs de la croissance du PIB.

### Les Données (L'Input)
* **Source :** Fichier `real-gdp-growth.csv`.
* **Nettoyage initial :** Suppression des entrées avec des valeurs cibles manquantes, ramenant l'ensemble de données à **8211 entrées**.

### Préparation des Caractéristiques (Data Wrangling)
1.  **Imputation :** Les valeurs manquantes dans la colonne `Code` (Code Pays) ont été imputées avec la valeur `Unknown`.
2.  **Encodage :** Les caractéristiques catégorielles (`Entity` et `Code`) ont été transformées via *One-Hot Encoding*, ce qui a entraîné l'ajout de **401 nouvelles colonnes**.
3.  **Mise à l'Échelle :** La colonne `Year` (Année) a été normalisée à l'aide d'un `StandardScaler`.

---

## 3. Analyse Approfondie : Modélisation

### Algorithmes Testés
Les modèles de régression suivants ont été évalués:
* Régression Linéaire (`Linear Regression`)
* Régression Ridge
* Régression Lasso
* Arbre de Décision (`Decision Tree`)
* **Forêt Aléatoire** (`Random Forest`)
* **Gradient Boosting**

### Résultats de Performance (Évaluation)
L'évaluation a révélé que tous les modèles ont obtenu des scores $\text{R}^2$ relativement faibles, suggérant que les caractéristiques actuelles n'expliquent qu'une partie de la variance de la croissance du PIB.

| Métrique | Forêt Aléatoire (Random Forest) | Gradient Boosting |
| :--- | :---: | :---: |
| **Coefficient de Détermination ($\text{R}^2$)** | **0.21** (Meilleur score) | 0.17 |
| **Erreur Quadratique Moyenne (MSE)** | 34.79 | 36.86 |
| **Erreur Absolue Moyenne (MAE)** | 3.04 | 3.23 |

* **Contre-Performance :** La Régression Lasso a montré la performance la plus faible avec un $\text{R}^2$ de -0.00, indiquant qu'elle est moins efficace que la simple prédiction de la moyenne.

### Conclusion sur l'Algorithme

Le modèle de **Régression par Forêt Aléatoire (Random Forest Regression)** a été identifié comme le meilleur performeur, bien que son pouvoir explicatif global (R² de 0.21) reste modeste.

---

## 4. Perspectives et Prochaines Étapes

Pour améliorer la performance des modèles et augmenter le score $\text{R}^2$, les actions futures doivent se concentrer sur l'optimisation et l'enrichissement des données :

1.  **Optimisation :** Réaliser un affinage des hyperparamètres des modèles **Random Forest** et **Gradient Boosting**.
2.  **Ingénierie de Caractéristiques :** Explorer l'ajout de variables externes pour mieux expliquer la variance du PIB, telles que :
    * Des indicateurs macroéconomiques supplémentaires.
    * Des événements géopolitiques majeurs.
    * Des caractéristiques temporelles plus granulaires.

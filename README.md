# Systèmes de Recommandation Durables — Mémoire de Master

**Auteur :** Charles Molle  
**Institution :** UCLouvain — Louvain School of Management  
**Diplôme :** Master [120] en Ingénieur de gestion, à finalité spécialisée — Majeure Business Analytics  
**Année académique :** 2025–2026  
**Promotrice :** Prof. Chloé Satinet  

---

## Question de recherche

*Dans quelle mesure l'adaptation du paramètre de pondération β au niveau 
utilisateur permet-elle d'améliorer le compromis entre précision et durabilité 
dans les stratégies de post-processing des systèmes de recommandation ?*

---

## Structure du repository
├── code/
│   ├── Thèse - Recommender system - Algorithms on DS1 (for TW).ipynb
│   └── prepare_DS2.py
├── results/
│   ├── results_all_DS1.csv
│   ├── results_all_DS2.csv
│   ├── results_content_DS1.csv
│   ├── results_content_DS2.csv
│   ├── results_itemknn_DS1.csv
│   ├── results_itemknn_DS2.csv
│   ├── results_knn_DS1.csv
│   ├── results_knn_DS2.csv
│   ├── results_mf_DS1.csv
│   └── results_mf_DS2.csv
└── data/
└── README.md

---

## Jeux de données

- **DS1** — Amazon Clothing, Shoes & Jewelry  
  Source : https://nijianmo.github.io/amazon/index.html

- **DS2** — Amazon Clothing  
  Source : https://amazon-reviews-2023.github.io/

> **Note :** Les jeux de données ne sont pas inclus dans ce repository en raison 
> de leur taille. Veuillez consulter les liens ci-dessus pour les télécharger.

---

## Dépendances

```bash
pip install numpy pandas scikit-learn scipy sentence-transformers matplotlib seaborn
```

---

## Note sur les chemins d'accès

Les chemins d'accès dans le code sont configurés pour l'environnement local 
de développement. Veuillez les adapter avant d'exécuter le code.

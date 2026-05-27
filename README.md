# Systèmes de Recommandation Durables — Mémoire de Master

**Auteur :** Charles Molle  
**Institution :** UCLouvain — Louvain School of Management  
**Diplôme :** Master [120] en Ingénieur de gestion, à finalité spécialisée — Majeure Business Analytics  
**Année académique :** 2025–2026  
**Promotrice :** Prof. Chloé Satinet  

---

## Question de recherche

*Dans quelle mesure l'adaptation du paramètre de pondération β au niveau utilisateur permet-elle d'améliorer le compromis entre précision et durabilité dans les stratégies de post-processing des systèmes de recommandation ?*

---

## Structure du repository

```
├── code_base/
│   └── Thèse - Recommender system - Algorithms on DS1 (for TW).ipynb
├── data/
│   ├── Dataset1p1.csv                         ← DS1 interactions (Amazon Clothing, Shoes & Jewelry)
│   ├── Dataset1p2.csv                         ← DS1 métadonnées items (avec Pred_EI)
│   ├── Dataset2p1.csv                         ← DS2 interactions (Amazon Clothing)
│   ├── Dataset2p2.csv                         ← DS2 métadonnées items (co2e_kg, sustainability_score)
│   ├── clothing_gemini.jsonl                  ← enrichissement co2e_kg DS2 via Gemini API
│   ├── embed_all-MiniLM-L6-v2_6723.npy       ← cache embeddings BERT DS1 (6 723 items)
│   ├── embed_all-MiniLM-L6-v2_7600.npy       ← cache embeddings BERT DS2 (7 600 items)
│   ├── amazon_clothing/                       ← données brutes DS2 (format RecBole)
│   │   ├── amazon_clothing.inter
│   │   ├── amazon_clothing.train.inter
│   │   ├── amazon_clothing.valid.inter
│   │   └── amazon_clothing.test.inter
│   └── prepare_DS2.py                         ← script de construction et nettoyage de DS2
└── results/
    ├── results_all_DS1.csv
    ├── results_all_DS2.csv
    ├── results_content_DS1.csv
    ├── results_content_DS2.csv
    ├── results_itemknn_DS1.csv
    ├── results_itemknn_DS2.csv
    ├── results_knn_DS1.csv
    ├── results_knn_DS2.csv
    ├── results_mf_DS1.csv
    └── results_mf_DS2.csv
```

---

## Jeux de données

Les données sont incluses dans ce repository.

- **DS1** — Amazon Clothing, Shoes & Jewelry (dataset principal, feedback explicite 1–5)  
  Source : [nijianmo.github.io/amazon](https://nijianmo.github.io/amazon/index.html)  
  Traitement : filtrage 5-core, encodage ordinal, prédiction d'impact environnemental via `Pred_EI` (1–5, 1 = plus durable)

- **DS2** — Amazon Clothing (dataset de validation, feedback implicite)  
  Source : [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)  
  Traitement : filtrage 12-core itératif (5 000 utilisateurs, seed=42), enrichissement `co2e_kg` via Gemini API, seuil de durabilité P33

---

## Résultats

Le dossier `results/` contient les fichiers CSV produits par les boucles d'évaluation du notebook (460 combinaisons : 4 modèles × 2 stratégies × 11 modes β × N=5 × 5 folds) pour DS1 et DS2.

---

## Dépendances

```bash
pip install numpy pandas scikit-learn scipy matplotlib xlsxwriter sentence-transformers
```

---

## Note sur les chemins d'accès

Les chemins d'accès dans le notebook sont configurés pour l'environnement local de développement (`/Users/charlesmolle/Desktop/memoire_reco/`). Veuillez les adapter avant d'exécuter le code.

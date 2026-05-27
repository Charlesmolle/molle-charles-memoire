"""
prepare_DS2.py — Préparation du dataset DS2 (Amazon Clothing)
=============================================================
Génère Dataset2p1.csv et Dataset2p2.csv depuis les fichiers bruts.

Sources
-------
- data/amazon_clothing/amazon_clothing.inter
    Interactions RecBole : user_id:token  item_id:token  rating:float
    Les item_id sont des entiers 0..N-1 encodés alphabétiquement sur les ASINs.

- data/clothing_gemini.jsonl
    Métadonnées : parent_asin, product_name, co2e_kg (estimé par LLM)

Pipeline
--------
1. Charger les interactions, convertir tous les ratings en 1 (feedback implicite)
2. Supprimer les doublons utilisateur-item
3. Appliquer un filtrage 12-core itératif (15-core vide le dataset, cf. README)
4. Échantillonner aléatoirement 5 000 utilisateurs (seed=42)
5. Reconstruire le mapping item_id → ASIN (tri alphabétique des ASINs)
6. Joindre co2e_kg et product_name depuis le JSONL
7. Calculer sustainability_score = 1 − (log(co2+1) − min_log) / (max_log − min_log)
8. Sauvegarder Dataset2p1.csv et Dataset2p2.csv

Paramètres reproductibles
--------------------------
K_CORE   = 12       (filtrage k-core)
N_SAMPLE = 5000     (nombre d'utilisateurs échantillonnés)
SEED     = 42       (graine aléatoire)

Seuil de durabilité (non appliqué ici, utilisé en évaluation)
--------------------------------------------------------------
CO2_THRESHOLD = 33e percentile (P33) de co2e_kg sur les items retenus après filtrage
              = 5.14 kg  (P33)
"""

import json
import os
import numpy as np
import pandas as pd

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INTER_PATH = os.path.join(BASE_DIR, "amazon_clothing", "amazon_clothing.inter")
JSONL_PATH = os.path.join(BASE_DIR, "clothing_gemini.jsonl")
OUT_P1     = os.path.join(BASE_DIR, "Dataset2p1.csv")
OUT_P2     = os.path.join(BASE_DIR, "Dataset2p2.csv")

# ── Paramètres ────────────────────────────────────────────────────────────────
K_CORE   = 12
N_SAMPLE = 5000
SEED     = 42

# =============================================================================
# Étape 1 — Chargement et pré-traitement des interactions
# =============================================================================
print("Étape 1 — Chargement des interactions...")
df = pd.read_csv(INTER_PATH, sep="\t")
df.columns = ["User_ID", "Item_ID", "Rating"]

# Tous les ratings → 1 : on modélise le comportement implicite (vu = consommé)
df["Rating"] = 1

# Supprimer les doublons (un utilisateur peut avoir plusieurs ratings pour le
# même item dans RecBole si l'item apparaît dans plusieurs partitions)
n_raw = len(df)
df = df.drop_duplicates(subset=["User_ID", "Item_ID"])
print(f"  Brut : {n_raw:,} | Après déduplication : {len(df):,} interactions")
print(f"  Utilisateurs : {df.User_ID.nunique():,} | Items : {df.Item_ID.nunique():,}")

# =============================================================================
# Étape 2 — Filtrage k-core itératif
# =============================================================================
# Objectif : garder uniquement les utilisateurs et items avec ≥ K_CORE interactions.
# Le filtrage est itératif car supprimer des utilisateurs peut faire passer certains
# items sous le seuil, et vice versa.
# Note : 15-core vide entièrement ce dataset (médiane = 9 interactions/user,
# seuls 16 % ont ≥ 15 interactions → cascade jusqu'à 0 ligne).
print(f"\nÉtape 2 — Filtrage {K_CORE}-core itératif...")
it = 0
while True:
    it += 1
    u_cnt = df.groupby("User_ID")["Item_ID"].count()
    i_cnt = df.groupby("Item_ID")["User_ID"].count()
    df_new = df[
        df.User_ID.isin(u_cnt[u_cnt >= K_CORE].index) &
        df.Item_ID.isin(i_cnt[i_cnt >= K_CORE].index)
    ]
    removed = len(df) - len(df_new)
    df = df_new
    if removed == 0:
        break
print(f"  Convergé en {it} itérations")
print(f"  Résultat : {df.User_ID.nunique():,} users | {df.Item_ID.nunique():,} items | {len(df):,} interactions")

# =============================================================================
# Étape 3 — Échantillonnage aléatoire de N_SAMPLE utilisateurs
# =============================================================================
# Le 12-core produit ~14 800 utilisateurs, trop grand pour être comparable à DS1
# (4 110 users). On sous-échantillonne aléatoirement pour obtenir ~5 000 users.
print(f"\nÉtape 3 — Échantillonnage {N_SAMPLE} utilisateurs (seed={SEED})...")
rng = np.random.default_rng(SEED)
sampled_users = rng.choice(df.User_ID.unique(), size=N_SAMPLE, replace=False)
df = df[df.User_ID.isin(sampled_users)].copy()
print(f"  Résultat : {df.User_ID.nunique():,} users | {df.Item_ID.nunique():,} items | {len(df):,} interactions")
density = len(df) / (df.User_ID.nunique() * df.Item_ID.nunique())
print(f"  Densité : {density:.4%}")

# =============================================================================
# Étape 4 — Mapping item_id (entier RecBole) → ASIN (string)
# =============================================================================
# RecBole encode les item_id en triant les ASINs alphabétiquement :
#   item_id 0 → sorted(all_asins)[0], item_id 1 → sorted(all_asins)[1], etc.
print("\nÉtape 4 — Reconstruction du mapping item_id → ASIN...")
records = []
with open(JSONL_PATH) as f:
    for line in f:
        d = json.loads(line)
        records.append({
            "parent_asin":  d["parent_asin"],
            "product_name": d["product_name"],
            "co2e_kg":      float(d["co2e_kg"]),
        })

asins_sorted = sorted(r["parent_asin"] for r in records)
id_to_asin   = {i: asin for i, asin in enumerate(asins_sorted)}
asin_to_meta = {r["parent_asin"]: r for r in records}

# Vérification : tous les item_ids doivent avoir un ASIN dans le mapping
items_final = sorted(df.Item_ID.unique())
missing = [iid for iid in items_final if iid not in id_to_asin]
if missing:
    raise ValueError(f"{len(missing)} item_ids sans ASIN dans le mapping — vérifier le JSONL")
print(f"  {len(items_final)} items → {len(items_final)} ASINs (correspondance 100 %)")

# =============================================================================
# Étape 5 — Construction du DataFrame métadonnées
# =============================================================================
print("\nÉtape 5 — Construction des métadonnées...")
meta_rows = []
for iid in items_final:
    asin = id_to_asin[iid]
    rec  = asin_to_meta[asin]
    meta_rows.append({
        "Item_ID":      asin,
        "Item_title":   rec["product_name"],
        "co2e_kg":      rec["co2e_kg"],
    })
df_meta = pd.DataFrame(meta_rows)

# =============================================================================
# Étape 6 — Score de durabilité normalisé (log + min-max)
# =============================================================================
# Transformation log pour compresser la longue queue droite (max = 217 kg).
# Formule : score = 1 − (log(co2+1) − min_log) / (max_log − min_log)
# score = 1 → item le plus durable (co2 minimal)
# score = 0 → item le moins durable (co2 maximal)
print("\nÉtape 6 — Calcul du sustainability_score (log-minmax)...")
co2     = df_meta["co2e_kg"].values
log_co2 = np.log(co2 + 1)
log_min = log_co2.min()
log_max = log_co2.max()
score   = 1.0 - (log_co2 - log_min) / (log_max - log_min)
df_meta["sustainability_score"] = np.round(score, 6)

co2_threshold = float(np.percentile(co2, 33))
n_sust        = (co2 <= co2_threshold).sum()
print(f"  co2e_kg : min={co2.min():.2f} | max={co2.max():.2f} | moy={co2.mean():.2f} | P33={co2_threshold:.2f} kg")
print(f"  log range : [{log_min:.4f}, {log_max:.4f}]")
print(f"  score range : [{score.min():.4f}, {score.max():.4f}]")
print(f"  Seuil durabilité = P33 co2e_kg = {co2_threshold:.2f} kg → score ≥ {1-(np.log(co2_threshold+1)-log_min)/(log_max-log_min):.4f}")
print(f"  Items durables (co2 ≤ P33) : {n_sust}/{len(co2)} ({n_sust/len(co2)*100:.1f}%)")

# =============================================================================
# Étape 7 — Remplacement item_id entier → ASIN dans les interactions
# =============================================================================
df["Item_ID"] = df["Item_ID"].map(id_to_asin)

# =============================================================================
# Étape 8 — Sauvegarde CSV (même format que Dataset1p1.csv / Dataset1p2.csv)
# =============================================================================
# Dataset2p1.csv : interactions
#   Colonnes : [index];Rating;User_ID;Item_ID
#   Rating = 1 pour toutes les interactions (feedback implicite)
#   User_ID = entier RecBole (sera ré-encodé 0..N-1 par OrdinalEncoder dans le notebook)
#   Item_ID = ASIN string (sera encodé par OrdinalEncoder en respectant l'ordre alphabétique)
print("\nÉtape 7 — Sauvegarde des CSV...")
df_p1 = df[["Rating", "User_ID", "Item_ID"]].reset_index(drop=True)
df_p1.to_csv(OUT_P1, sep=";")
print(f"  {OUT_P1} — {len(df_p1):,} lignes")

# Dataset2p2.csv : métadonnées items
#   Colonnes : [index];Item_ID;Item_title;co2e_kg;sustainability_score
#   Pas de colonne Pred_EI (co2e_kg est continu, non discrétisé)
#   sustainability_score ∈ [0,1] : 1=plus durable, 0=moins durable
df_p2 = df_meta[["Item_ID", "Item_title", "co2e_kg", "sustainability_score"]].reset_index(drop=True)
df_p2.to_csv(OUT_P2, sep=";")
print(f"  {OUT_P2} — {len(df_p2):,} lignes")

# =============================================================================
# Résumé
# =============================================================================
print("\n" + "=" * 60)
print("RÉSUMÉ DS2 (Amazon Clothing)")
print("=" * 60)
print(f"  Filtrage         : {K_CORE}-core itératif")
print(f"  Échantillonnage  : {N_SAMPLE} users, seed={SEED}")
print(f"  Utilisateurs     : {df.User_ID.nunique():,}")
print(f"  Items            : {len(df_p2):,}")
print(f"  Interactions     : {len(df_p1):,}")
print(f"  Densité          : {density:.4%}")
print(f"  Seuil durabilité : co2e_kg ≤ {co2_threshold:.2f} kg (P33)")
print(f"  Items durables   : {n_sust}/{len(co2)} ({n_sust/len(co2)*100:.1f}%)")
print(f"  Dataset2p1.csv   : {OUT_P1}")
print(f"  Dataset2p2.csv   : {OUT_P2}")

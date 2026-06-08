#!/usr/bin/env python3
"""
predict_match.py — Prevê a probabilidade de vitória entre dois jogadores de tênis.
Uso: python predict_match.py
"""

import difflib
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE     = os.path.join(os.path.dirname(__file__),
                        'new-dataset', 'tennis_MatchChartingProject-master')
DATASET  = os.path.join(BASE, 'tennis_final_dataset.csv')
PROFILES = os.path.join(BASE, 'tennis_player_latest_profiles.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lr_charting.pkl')

PLAYER_FEATS = [
    'avg_first_serve_pct', 'avg_first_serve_won_pct', 'avg_second_serve_won_pct',
    'avg_ace_pct', 'avg_df_pct', 'avg_return_won_pct', 'avg_winners_per_pt',
    'avg_ue_per_pt', 'avg_winners_fh_ratio', 'avg_bp_save_pct',
    'avg_bp_clutch_save_pct', 'avg_bp_first_in_pct', 'avg_bpo_conv_pct',
    'avg_bpo_ue_pct', 'avg_srv_win_1_3', 'avg_srv_win_4_6', 'avg_srv_win_7plus',
    'avg_ret_win_1_3', 'avg_ret_win_4_6', 'avg_ret_win_7plus',
    'elo', 'elo_surface',
]
DIFF_COLS = [f'diff_{f}' for f in PLAYER_FEATS]
CAT_COLS  = ['Surface', 'Round', 'Best of']


# ─── Treinamento ──────────────────────────────────────────────────────────────

def _build_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[CAT_COLS].copy()
    for feat in PLAYER_FEATS:
        out[f'diff_{feat}'] = df[f'p1_{feat}'] - df[f'p2_{feat}']
    return out


def train_and_save() -> tuple:
    print('Treinando modelo (primeira execução)...')
    df = pd.read_csv(DATASET)
    y  = df['player_1_wins?']
    X  = _build_diff_features(df)

    encoders: dict[str, LabelEncoder] = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler()),
        ]), DIFF_COLS),
        ('cat', SimpleImputer(strategy='most_frequent'), CAT_COLS),
    ])

    pipeline = Pipeline([
        ('prep',  preprocessor),
        ('model', LogisticRegression(C=0.1, solver='lbfgs',
                                     max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'pipeline': pipeline, 'encoders': encoders}, f)

    print(f'Modelo salvo em {MODEL_PATH}\n')
    return pipeline, encoders


def _load_or_train() -> tuple:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        return data['pipeline'], data['encoders']
    return train_and_save()


# ─── Busca de jogadores ───────────────────────────────────────────────────────

def _find_player(name: str, all_players: list) -> list:
    name_lower = name.lower()
    partial = [p for p in all_players if name_lower in p.lower()]
    if partial:
        return partial[:8]
    return difflib.get_close_matches(name, all_players, n=8, cutoff=0.35)


def _select_player(prompt: str, profiles: pd.DataFrame) -> pd.Series:
    all_players = profiles['player'].tolist()
    while True:
        name = input(prompt).strip()
        if not name:
            continue
        candidates = _find_player(name, all_players)
        if not candidates:
            print('  Nenhum jogador encontrado. Tente parte do sobrenome.\n')
            continue
        if len(candidates) == 1:
            chosen = candidates[0]
            print(f'  → {chosen}')
            return profiles.loc[profiles['player'] == chosen].iloc[0]
        print('  Jogadores encontrados:')
        for i, c in enumerate(candidates, 1):
            print(f'    [{i}] {c}')
        choice = input('  Escolha o número: ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            chosen = candidates[int(choice) - 1]
            return profiles.loc[profiles['player'] == chosen].iloc[0]
        print('  Opção inválida.\n')


# ─── Previsão ─────────────────────────────────────────────────────────────────

def _predict(p1: pd.Series, p2: pd.Series,
             surface: str, round_: str, best_of: int,
             pipeline, encoders: dict) -> tuple:
    row = {f'diff_{f}': float(p1.get(f, np.nan)) - float(p2.get(f, np.nan))
           for f in PLAYER_FEATS}
    row['Surface'] = surface
    row['Round']   = round_
    row['Best of'] = best_of

    X = pd.DataFrame([row])
    for col in CAT_COLS:
        le  = encoders[col]
        val = str(X[col].iloc[0])
        X[col] = le.transform([val]) if val in le.classes_ else [0]

    proba = pipeline.predict_proba(X)[0]
    return float(proba[1]), float(proba[0])


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _ask_surface(encoders: dict) -> str:
    known = sorted(encoders['Surface'].classes_)
    print(f'\n  Superfícies disponíveis: {", ".join(known)}')
    surf = input('  Superfície [Hard]: ').strip() or 'Hard'
    if surf not in known:
        print(f'  "{surf}" não reconhecida; usando "Hard".')
        surf = 'Hard'
    return surf


def main():
    pipeline, encoders = _load_or_train()
    profiles = pd.read_csv(PROFILES)

    print('╔══════════════════════════════════════╗')
    print('║   Previsão de Partida de Tênis  🎾   ║')
    print('╚══════════════════════════════════════╝\n')

    p1 = _select_player('Jogador 1 (nome ou sobrenome): ', profiles)
    p2 = _select_player('Jogador 2 (nome ou sobrenome): ', profiles)
    surface = _ask_surface(encoders)

    p1_prob, p2_prob = _predict(p1, p2, surface, 'F', 3, pipeline, encoders)

    line = '─' * 46
    print(f'\n{line}')
    print(f'  {"Jogador":<32} {"Prob. vitória":>12}')
    print(line)
    print(f'  {p1["player"]:<32} {p1_prob*100:>11.1f}%')
    print(f'  {p2["player"]:<32} {p2_prob*100:>11.1f}%')
    print(line)

    winner = p1["player"] if p1_prob >= p2_prob else p2["player"]
    elo1   = p1.get('elo', 1500)
    elo2   = p2.get('elo', 1500)
    diff   = float(elo1) - float(elo2)

    print(f'\n  Favorito: {winner}')
    print(f'  ELO {p1["player"].split()[0]}: {float(elo1):.0f}  |  '
          f'ELO {p2["player"].split()[0]}: {float(elo2):.0f}  |  '
          f'Diferença: {diff:+.0f}')
    print(f'  Superfície: {surface}\n')


if __name__ == '__main__':
    main()

# Tennis Machine Learning

Projeto acadêmico de mineração de dados para previsão de vencedores em partidas de tênis masculino (ATP), usando o *Tennis Match Charting Project* como fonte de dados.

## Visão Geral

**Problema:** prever o vencedor de uma partida antes de ela acontecer, com base no histórico de desempenho acumulado dos dois jogadores.

**Melhor modelo:** Regressão Logística — F1 ≈ 0,67 em validação cruzada (5-fold).

**Feature mais preditiva:** diferença de rating ELO entre os jogadores (correlação 0,407 com o resultado).

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Previsão interativa

Informe dois jogadores e o script retorna a probabilidade de vitória de cada um:

```bash
python predict_match.py
```

**Exemplo de uso:**

```
╔══════════════════════════════════════╗
║   Previsão de Partida de Tênis  🎾   ║
╚══════════════════════════════════════╝

Jogador 1 (nome ou sobrenome): Djokovic
  → Novak Djokovic
Jogador 2 (nome ou sobrenome): Alcaraz
  → Carlos Alcaraz

  Superfícies disponíveis: Carpet, Clay, Grass, Hard
  Superfície [Hard]: Clay

──────────────────────────────────────────────
  Jogador                            Prob. vitória
──────────────────────────────────────────────
  Novak Djokovic                           54.3%
  Carlos Alcaraz                           45.7%
──────────────────────────────────────────────

  Favorito: Novak Djokovic
  ELO Novak: 1924  |  ELO Carlos: 1832  |  Diferença: +92
  Superfície: Clay
```

> **Primeira execução:** o modelo é treinado automaticamente a partir do dataset e salvo em `models/lr_charting.pkl`. As execuções seguintes carregam o modelo salvo.

---

## Estrutura do projeto

```
tennis-machine-learning/
├── predict_match.py                   # Script de previsão interativa
├── requirements.txt                   # Dependências Python
├── relatorio.md                       # Relatório acadêmico completo
├── README.md
├── models/
│   └── lr_charting.pkl                # Modelo salvo (gerado na 1ª execução)
└── new-dataset/
    └── tennis_MatchChartingProject-master/
        ├── matches.csv                          # 7.569 partidas com metadados
        ├── charting-m-stats-Overview.csv        # Estatísticas gerais por partida
        ├── charting-m-stats-Rally.csv           # Desempenho por comprimento de rally
        ├── charting-m-stats-KeyPointsServe.csv  # Break points sofridos
        ├── charting-m-stats-KeyPointsReturn.csv # Oportunidades de quebra
        ├── charting-m-points-*.csv              # Histórico ponto a ponto
        ├── tennis_player_latest_profiles.csv    # Último perfil de cada jogador
        └── tennis_final_dataset.csv             # Dataset final para modelagem
└── tennis_ml/
    └── notebooks/
        ├── 01_extract_overview.ipynb    # Features de serviço e retorno
        ├── 02_extract_rally.ipynb       # Features de rally
        ├── 03_extract_keypoints.ipynb   # Features de pontos decisivos
        ├── 04_creating_winner_dataset.ipynb  # Extração do alvo
        ├── 05_creating_profile_dataset.ipynb # Perfis + ELO
        ├── 06_improving_matches_dataset.ipynb # Dataset final + diffs
        ├── 07_eda.ipynb                 # Análise exploratória
        └── 08_modelling.ipynb           # Treinamento e avaliação
```

---

## Pipeline de notebooks

| Notebook | O que faz | Saída |
|---|---|---|
| 01 | Extrai ratios de serviço/retorno e médias acumuladas | `features_overview.csv` |
| 02 | Extrai eficiência por comprimento de rally | `features_rally.csv` |
| 03 | Extrai desempenho em break points e game points | `features_keypoints.csv` |
| 04 | Determina o vencedor de cada partida pelo último ponto | `winners.csv` |
| 05 | Consolida perfis por jogador + ratings ELO global e por superfície | `tennis_player_profiles.csv` |
| 06 | Monta dataset de partidas com features de ambos os jogadores e cria diffs | `tennis_final_dataset.csv` |
| 07 | Análise exploratória: correlações, nulos, distribuições | — |
| 08 | Treina e compara Benchmark ELO, XGBoost, MLP e Regressão Logística | — |

---

## Features do modelo

O modelo usa **28 features**: 22 diferenças numéricas (`diff_feat = p1_feat − p2_feat`) + 3 categóricas.

### Diferenças numéricas (22)
| Grupo | Features |
|---|---|
| Serviço geral | `diff_avg_first_serve_pct`, `diff_avg_first_serve_won_pct`, `diff_avg_second_serve_won_pct`, `diff_avg_ace_pct`, `diff_avg_df_pct` |
| Retorno | `diff_avg_return_won_pct` |
| Qualidade de jogo | `diff_avg_winners_per_pt`, `diff_avg_ue_per_pt`, `diff_avg_winners_fh_ratio`, `diff_avg_bp_save_pct` |
| Pontos decisivos | `diff_avg_bp_clutch_save_pct`, `diff_avg_bp_first_in_pct`, `diff_avg_bpo_conv_pct`, `diff_avg_bpo_ue_pct` |
| Rally | `diff_avg_srv_win_1_3/4_6/7plus`, `diff_avg_ret_win_1_3/4_6/7plus` |
| Ratings ELO | `diff_elo`, `diff_elo_surface` |

### Categóricas (3)
`Surface`, `Round`, `Best of`

---

## Resultados dos modelos (validação cruzada)

| Modelo | F1 (CV) | Observação |
|---|---|---|
| Benchmark ELO | ~0,665 | Regra determinística |
| XGBoost | 0,661 ± 0,008 | Gradient boosting |
| MLP | 0,664 ± 0,006 | 3 camadas (128–64–32) |
| **Regressão Logística** | **0,669 ± 0,010** | Melhor resultado |

---

## Fonte de dados

**Tennis Match Charting Project** — base colaborativa com estatísticas ponto a ponto de partidas ATP.
Repositório original: [tennis_MatchChartingProject](https://github.com/JeffSackmann/tennis_MatchChartingProject)

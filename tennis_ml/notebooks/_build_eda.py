import json

path = r'c:\Users\lucas\Desktop\Projetos\Faculdade\Tennis\tennis-machine-learning\tennis_ml\notebooks\eda.ipynb'

def md(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [source]}

def code(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}

base_cells = [
    md('# EDA - Tennis Match Charting Project\nDataset: charting-m-matches.csv (masculino)'),
    code([
        'import pandas as pd\n',
        'import matplotlib.pyplot as plt\n',
        'import seaborn as sns\n',
        '\n',
        'BASE_PATH = "../../new-dataset/tennis_MatchChartingProject-master"\n',
        '\n',
        'df_men = pd.read_csv(f"{BASE_PATH}/charting-m-matches.csv")\n',
        'df_women = pd.read_csv(f"{BASE_PATH}/charting-w-matches.csv")\n',
        '\n',
        'print(f"Masculino: {df_men.shape}")\n',
        'print(f"Feminino:  {df_women.shape}")',
    ]),

    md('## Head'),
    code(['df_men.head()']),

    md('## Info'),
    code(['df_men.info()']),

    md('## Describe'),
    code(['df_men.describe(include="all")']),

    md('## Valores nulos'),
    code([
        'nulls = df_men.isnull().sum().sort_values(ascending=False)\n',
        'null_pct = (nulls / len(df_men) * 100).round(2)\n',
        'null_df = nulls[nulls > 0].to_frame(name="count")\n',
        'null_df["pct"] = null_pct[null_pct > 0]\n',
        'print(null_df)\n',
        '\n',
        'plt.figure(figsize=(8, 4))\n',
        'sns.barplot(x=null_df.index, y=null_df["pct"])\n',
        'plt.title("% de valores nulos por coluna (masculino)")\n',
        'plt.ylabel("%")\n',
        'plt.xticks(rotation=45, ha="right")\n',
        'plt.tight_layout()\n',
        'plt.show()',
    ]),

    md('## Distribuicao temporal (Year)'),
    code([
        'df_men["Date_parsed"] = pd.to_datetime(df_men["Date"], format="%Y%m%d", errors="coerce")\n',
        'df_men["Year"] = df_men["Date_parsed"].dt.year\n',
        '\n',
        'print(f"Periodo: {df_men[\'Year\'].min():.0f} - {df_men[\'Year\'].max():.0f}")\n',
        'print(f"Datas invalidas: {df_men[\'Year\'].isna().sum()}")\n',
        '\n',
        'plt.figure(figsize=(12, 4))\n',
        'df_men["Year"].value_counts().sort_index().plot(kind="bar")\n',
        'plt.title("Partidas por ano (masculino)")\n',
        'plt.xlabel("Ano")\n',
        'plt.ylabel("Qtd")\n',
        'plt.tight_layout()\n',
        'plt.show()',
    ]),

    md('## Surface'),
    code([
        'print(df_men["Surface"].value_counts())\n',
        '\n',
        'plt.figure(figsize=(6, 4))\n',
        'df_men["Surface"].value_counts().plot(kind="bar")\n',
        'plt.title("Distribuicao por superficie (masculino)")\n',
        'plt.ylabel("Qtd")\n',
        'plt.xticks(rotation=0)\n',
        'plt.tight_layout()\n',
        'plt.show()',
    ]),

    md('## Round'),
    code([
        'round_order = ["Q1","Q2","Q3","R128","R64","R32","R16","QF","SF","F","BR","RR","PO","PQ"]\n',
        'round_counts = df_men["Round"].value_counts()\n',
        'print(round_counts)\n',
        '\n',
        'plt.figure(figsize=(10, 4))\n',
        'ordered = [r for r in round_order if r in round_counts.index]\n',
        'others = [r for r in round_counts.index if r not in round_order]\n',
        'round_counts[ordered + others].plot(kind="bar")\n',
        'plt.title("Distribuicao por round (masculino)")\n',
        'plt.ylabel("Qtd")\n',
        'plt.xticks(rotation=45, ha="right")\n',
        'plt.tight_layout()\n',
        'plt.show()',
    ]),

    md('## Best of'),
    code([
        'print(df_men["Best of"].value_counts())\n',
        '\n',
        'plt.figure(figsize=(4, 3))\n',
        'df_men["Best of"].value_counts().sort_index().plot(kind="bar")\n',
        'plt.title("Best of (masculino)")\n',
        'plt.ylabel("Qtd")\n',
        'plt.xticks(rotation=0)\n',
        'plt.tight_layout()\n',
        'plt.show()',
    ]),

    md('## Lateralidade (Pl 1 hand / Pl 2 hand)'),
    code([
        'fig, axes = plt.subplots(1, 2, figsize=(8, 3))\n',
        'df_men["Pl 1 hand"].value_counts().plot(kind="bar", ax=axes[0], title="Pl 1 hand")\n',
        'df_men["Pl 2 hand"].value_counts().plot(kind="bar", ax=axes[1], title="Pl 2 hand")\n',
        'for ax in axes:\n',
        '    ax.set_xlabel("")\n',
        '    ax.tick_params(axis="x", rotation=0)\n',
        'plt.tight_layout()\n',
        'plt.show()\n',
        '\n',
        'df_men["Handedness_Matchup"] = df_men["Pl 1 hand"].fillna("U") + df_men["Pl 2 hand"].fillna("U")\n',
        'print(df_men["Handedness_Matchup"].value_counts())',
    ]),

    md('## Top 20 torneios'),
    code([
        'top20 = df_men["Tournament"].value_counts().head(20)\n',
        'print(top20)\n',
        '\n',
        'plt.figure(figsize=(10, 5))\n',
        'top20.plot(kind="barh")\n',
        'plt.title("Top 20 torneios (masculino)")\n',
        'plt.xlabel("Qtd de partidas")\n',
        'plt.gca().invert_yaxis()\n',
        'plt.tight_layout()\n',
        'plt.show()',
    ]),

    md('## Final TB? e Has_Umpire'),
    code([
        'print("Final TB?:")\n',
        'print(df_men["Final TB?"].value_counts(dropna=False))\n',
        'print()\n',
        'has_umpire = (df_men["Umpire"].notna() & (df_men["Umpire"] != "")).sum()\n',
        'print(f"Com umpire:  {has_umpire} ({has_umpire/len(df_men)*100:.1f}%)")\n',
        'print(f"Sem umpire:  {len(df_men)-has_umpire} ({(len(df_men)-has_umpire)/len(df_men)*100:.1f}%)")',
    ]),
]

nb = {
    'cells': base_cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'OK - {len(nb["cells"])} celulas')

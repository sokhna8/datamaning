# app.py ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import ast, base64, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster        import KMeans
from sklearn.metrics        import silhouette_score, adjusted_rand_score
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import plotly.express as px

# ─────────────────── Config & fond d’écran ──────────────────────────
st.set_page_config(page_title="E-commerce Analytics", layout="wide")

def set_background(image_file="alimentation.jpg"):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover; background-position: center;
            background-repeat: no-repeat; color: white;
        }}
        .stApp::before {{
            content:''; position:fixed; inset:0;
            background:rgba(0,0,0,.55); z-index:-1;
        }}
        </style>
        """,
        unsafe_allow_html=True)
set_background()

# ─────────────────── Menu latéral ────────────────────────────────────
st.sidebar.title("🧪 Menu")
page       = st.sidebar.radio("Choisir une page", 
                              ["🏠 Accueil", "Association", "K-means", "RFM"])
uploaded   = st.sidebar.file_uploader("📂 Charger un CSV", type="csv")
st.sidebar.markdown("---")

# Paramètres K-means (affichés uniquement si page = K-means)
if page == "K-means":
    use_pca  = st.sidebar.checkbox("Réduction PCA", value=True)
    pca_comp = st.sidebar.slider("Nb composantes", 2, 5, 2) if use_pca else None
    auto_k   = st.sidebar.checkbox("k automatique (Silhouette)", value=True)
    manual_k = st.sidebar.slider("k manuel", 2, 10, 4) if not auto_k else None
    n_init   = st.sidebar.number_input("Ré-initialisations ARI", 5, 50, 10)

# Si aucun fichier → page d’accueil seulement
if uploaded is None and page != "🏠 Accueil":
    st.info("Déposez d’abord un fichier CSV pour accéder aux analyses.")
    st.stop()

# ─────────────────── Fonctions utilitaires ──────────────────────────
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file, encoding="ISO-8859-1")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

@st.cache_data
def load_rules():
    rules = pd.read_csv("association_rules.csv")
    for col in ["antecedents", "consequents"]:
        rules[col] = rules[col].apply(ast.literal_eval)
    return rules

def preprocess_kmeans(df):
    df = df.copy()
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    agg = (df.groupby("customerid")
             .agg(nb_commandes    = ("invoiceno", "count"),
                  quantite_totale = ("quantity",  "sum"),
                  prix_moyen      = ("unitprice", "mean"),
                  nb_jours_achat  = ("invoicedate", lambda x: x.nunique()))
             .dropna()
             .reset_index())
    return agg

def choose_k(X):
    best_k, best_score = 2, -1
    for k in range(2,11):
        s = silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X))
        if s > best_score: best_k, best_score = k, s
    return best_k, best_score

def stability_ari(X, k, n=10):
    base = KMeans(k, n_init=1, random_state=0).fit(X).labels_
    aris = [adjusted_rand_score(base,
            KMeans(k, n_init=1, random_state=r).fit(X).labels_) for r in range(1,n)]
    return np.mean(aris)

# ─────────────────── Pages ──────────────────────────────────────────
def page_home():
    st.title("🏠 Accueil")
    st.markdown("""
    # 👋Bienvenue dans l'application d'analyse e-commerce!

    ###  Cette application interactive vous permet d'explorer différentes techniques de **Data Mining**
    appliquées à un jeu de données e-commerce :

                 Explorez :
    - 📦 Segmentation clients **K-means**
    - 🛍️ Règles d'association **FP-Growth**
    - 📈 Segmentation clients **RFM** 

                Sélectionnez un onglet dans la barre latérale.
    """)

def page_association():
    st.title("🛍️ Recommandation d'articles")
    rules = load_rules()

    unique_items = sorted({item for ant in rules['antecedents'] for item in ant})
    prod = st.selectbox("Choisir un produit", unique_items)

    min_conf = st.slider("Confiance min.", 0.0, 1.0, 0.5, 0.05)
    min_lift = st.slider("Lift min.",       0.0, 10., 1.0, 0.1)

    filt = rules[
        rules['antecedents'].apply(lambda x: prod in x) &
        (rules['confidence'] >= min_conf) &
        (rules['lift']       >= min_lift) ]

    filt = filt.sort_values("confidence", ascending=False)\
               .drop_duplicates(subset="consequents")

    if filt.empty:
        st.warning("Aucune recommandation ; baissez les seuils.")
    else:
        st.markdown(f"### 🎯 Recommandations pour **{prod}**")
        for _, r in filt.iterrows():
            st.markdown(f"- **{', '.join(r['consequents'])}**  \t"
                        f"*conf.* {r['confidence']:.2f} – *lift* {r['lift']:.2f}")

def page_kmeans():
    st.title("📊 Segmentation K-means")
    df_raw = load_csv(uploaded)
    df_cli = preprocess_kmeans(df_raw)
    st.write(f"**{len(df_cli)}** clients agrégés.")
    numeric = ['nb_commandes','quantite_totale','prix_moyen','nb_jours_achat']

    # scaling
    X = StandardScaler().fit_transform(df_cli[numeric])

    # PCA
    if use_pca:
        pca = PCA(n_components=pca_comp, random_state=0)
        Xred = pca.fit_transform(X)
        st.caption(f"Variance expliquée : {pca.explained_variance_ratio_.sum()*100:.1f} %")
    else:
        Xred = X

    # choix k
    if auto_k:
        k, sil = choose_k(Xred)
        st.caption(f"k optimal (Silhouette) : **{k}**  *(score {sil:.2f})*")
    else:
        k = manual_k

    km = KMeans(k, n_init=20, random_state=0).fit(Xred)
    df_cli["cluster"] = km.labels_

    # stabilité
    ari = stability_ari(X, k, n_init)
    st.caption(f"Stabilité moyenne ARI : **{ari:.3f}**")

    # résumé & description
    st.subheader("Moyennes par cluster")
    summary = (df_cli.groupby("cluster")[numeric]
                      .mean().round(2)
                      .assign(nb_clients=df_cli["cluster"].value_counts()))
    st.dataframe(summary)

    st.subheader("Description automatique")
    mean = summary.mean()
    desc=[]
    for cid,row in summary.iterrows():
        d  = f"### Cluster {cid}\n"
        d += "• Commandes fréquentes\n"       if row['nb_commandes']    > mean['nb_commandes']    else "• Commandes moins fréquentes\n"
        d += "• Quantité élevée\n"           if row['quantite_totale'] > mean['quantite_totale'] else "• Petite quantité\n"
        d += "• Panier moyen élevé\n"        if row['prix_moyen']      > mean['prix_moyen']      else "• Panier moyen faible\n"
        d += "• Achats récents\n"            if row['nb_jours_achat']  < mean['nb_jours_achat']  else "• Achats plus anciens\n"
        d += f"*{int(row['nb_clients'])} clients*"
        desc.append(d)
    st.markdown("\n\n".join(desc))

    # scatter
    if Xred.shape[1] >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=Xred[:,0], y=Xred[:,1], hue=df_cli["cluster"], palette="tab10", ax=ax)
        ax.set(xlabel="PC1", ylabel="PC2")
        st.pyplot(fig)

    # download

# ─────────────────── RFM ────────────────────────────────────────────
def page_rfm() -> None:
    st.title("📦 Segmentation RFM")

    # 1) Chargement & nettoyage de base
    df = load_csv(uploaded)                                 
    df = df[pd.notnull(df['customerid'])]
    df = df[(df['quantity'] > 0) & (df['unitprice'] > 0)]
    df = df.drop_duplicates()

    df['invoicedate']  = pd.to_datetime(df['invoicedate'])
    df['total_amount'] = df['quantity'] * df['unitprice']
    df['month']        = df['invoicedate'].dt.month

    # 2) Statistiques descriptives rapides
    st.subheader("Ventes par pays")
    pays = (df.groupby('country')['total_amount']
              .sum()
              .sort_values(ascending=False)
              .head(10))
    st.plotly_chart(px.bar(pays, title="Top 10 pays par chiffre d’affaires"))

    st.subheader("Ventes mensuelles")
    mois = df.groupby('month')['total_amount'].sum()
    st.plotly_chart(px.bar(mois, title="CA par mois"))

    # 3) Calcul des indicateurs R-F-M
    snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)

    rfm = (df.groupby('customerid')
             .agg(recency      = ('invoicedate', lambda x: (snapshot_date - x.max()).days),
                  frequency    = ('invoiceno',   'nunique'),
                  monetary     = ('total_amount','sum'))
           )

    # 4) Normalisation & clustering
    scaler        = StandardScaler()
    rfm_scaled    = scaler.fit_transform(rfm)

    n_clusters    = st.slider("Nombre de clusters K-means", 2, 10, 4)
    kmeans        = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # 5) Affichage résultat
    st.subheader("Aperçu des clients ")
    st.dataframe(rfm.head())

    # 2D
    st.subheader("Dispersion Recency vs Monetary (couleur = cluster)")
    st.plotly_chart(
        px.scatter(rfm,
                   x='recency', y='monetary',
                   color=rfm['cluster'].astype(str),
                   labels={'color':'cluster'})
    )

    # 3D
    st.subheader("Dispersion 3D R-F-M")
    st.plotly_chart(
        px.scatter_3d(rfm,
                      x='recency', y='frequency', z='monetary',
                      color=rfm['cluster'].astype(str),
                      labels={'color':'cluster'})
    )

    # 6) Export
    csv = rfm.reset_index().to_csv(index=False).encode()
    st.download_button("📥 Télécharger les résultats CSV",
                       data      = csv,
                       file_name = "segmentation_rfm_clusters.csv",
                       mime      = "text/csv")

# ─────────────────── Router ─────────────────────────────────────────
if page == "🏠 Accueil":
    page_home()
elif page == "Association":
    page_association()
elif page == "K-means":
    page_kmeans()
else:
    page_rfm()

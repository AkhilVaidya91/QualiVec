import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from kneed import KneeLocator
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from wordcloud import WordCloud
import re
import io
import base64

st.set_page_config(page_title="Sentence Clustering & Topic Explorer", layout="wide")

# ---------------------- Helpers ----------------------
@st.cache_resource
def load_embedding_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_data
def preprocess_for_topic_modeling(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = simple_preprocess(text, deacc=True)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def compute_embeddings(model, sentences, batch_size=64):
    return model.encode(sentences, show_progress_bar=True, batch_size=batch_size)

def find_optimal_k(embeddings, max_k=20):
    n = embeddings.shape[0]
    max_k = min(max_k, max(2, n // 2))
    k_range = list(range(2, max_k + 1))
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(embeddings, labels))
    # Knee detection
    knee = None
    try:
        knee_locator = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
        knee = knee_locator.knee
    except Exception:
        knee = None
    # Fallback: if knee is None choose k with max silhouette
    if knee is None:
        knee = k_range[int(np.argmax(silhouettes))]
    return knee, pd.DataFrame({"k": k_range, "inertia": inertias, "silhouette": silhouettes})

def run_kmeans(embeddings, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return km, labels

# Topic modeling per cluster; returns lda model, dictionary, corpus and topic assignments
def topic_model_for_cluster(sentences, num_topics=2, passes=10):
    processed = [preprocess_for_topic_modeling(s) for s in sentences]
    processed = [p for p in processed if len(p) > 0]
    if len(processed) < 3:
        return None
    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in processed]
    if len(dictionary) < 3:
        return None
    ntopics = min(num_topics, max(2, len(processed) // 2), max(2, len(dictionary) // 2))
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=ntopics,
                          random_state=42, passes=passes, alpha='auto')
    # compute dominant topic per original sentence
    doc_topics = []
    for bow in corpus:
        topics = lda.get_document_topics(bow)
        if len(topics) == 0:
            doc_topics.append((None, 0.0))
        else:
            # pick topic with highest weight
            top = max(topics, key=lambda x: x[1])
            doc_topics.append(top)
    return {
        'lda': lda,
        'dictionary': dictionary,
        'corpus': corpus,
        'processed': processed,
        'doc_topics': doc_topics
    }

# Utility: create a DataFrame with cluster, topic, sentence rows
def build_cluster_topic_table(df, cluster_col='cluster', sentence_col='sentence', per_cluster_topics=None):
    rows = []
    # per_cluster_topics: dict cluster_id -> topic_model_result
    for cluster_id, res in per_cluster_topics.items():
        if res is None:
            continue
        lda = res['lda']
        dictionary = res['dictionary']
        processed = res['processed']
        doc_topics = res['doc_topics']
        # build topic names
        topic_names = {}
        for tid in range(lda.num_topics):
            terms = lda.show_topic(tid, topn=6)
            topic_names[tid] = ", ".join([t for t, w in terms])
        # Map back to original sentences in that cluster
        cluster_sentences = df[df[cluster_col] == cluster_id][sentence_col].tolist()
        # Need to align processed docs with original sentences: some processed docs were dropped -> build index mapping
        proc_index = 0
        for orig in cluster_sentences:
            proc = preprocess_for_topic_modeling(orig)
            if len(proc) == 0:
                # assign None topic
                rows.append((cluster_id, 'NO_TOPIC', orig))
                continue
            bow = dictionary.doc2bow(proc)
            topics = lda.get_document_topics(bow)
            if len(topics) == 0:
                rows.append((cluster_id, 'NO_TOPIC', orig))
            else:
                top = max(topics, key=lambda x: x[1])
                tid, score = top
                tname = topic_names.get(tid, f'Topic {tid}')
                rows.append((cluster_id, f'Topic {tid}: {tname}', orig))
            proc_index += 1
    result_df = pd.DataFrame(rows, columns=['cluster', 'topic', 'sentence'])
    return result_df

# ---------------------- Streamlit UI ----------------------
st.title("🔎 Sentence Clustering & Topic Explorer")
st.markdown(
    "Upload a CSV with a column of sentences (default column name: `sentence`). The app will embed the sentences, find an optimal k, cluster them, visualize, and run per-cluster topic modeling."
)

with st.sidebar:
    st.header("Configuration")
    upload = st.file_uploader("Upload CSV", type=['csv'])
    example = st.checkbox("Use example demo data (small)")
    sentence_col = st.text_input("Sentence column name", value='sentence')
    embedding_model_name = st.text_input("SentenceTransformer model", value='all-MiniLM-L6-v2')
    max_k = st.number_input("Max k for search", min_value=4, max_value=200, value=20, step=1)
    run_tsne = st.checkbox("Run t-SNE (only for <= 5000 rows)", value=True)
    lda_topics_per_cluster = st.number_input("Max topics per cluster (suggested)", min_value=1, max_value=10, value=3)
    lda_passes = st.number_input("LDA passes", min_value=1, max_value=50, value=10)
    samples_per_topic = st.number_input("Example sentences per topic (rows limit)", min_value=1, max_value=20, value=3)

# Example data (small) — only used if user checks example
if example:
    demo_sentences = [
        'The cat sat on the mat.',
        'A dog was playing in the park.',
        'Quantum computing is an emerging field.',
        'Superconductors have zero resistance at low temperatures.',
        'I love reading historical novels.',
        'The stock market soared yesterday.',
        'Rainy days make me want to drink tea.',
        'Neural networks can approximate functions.',
        'Convolutional networks are good for images.',
        'This restaurant serves great sushi.'
    ]
    df = pd.DataFrame({sentence_col: demo_sentences})
else:
    df = None
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

if df is not None:
    if sentence_col not in df.columns:
        st.error(f"Column '{sentence_col}' not found in uploaded CSV")
    else:
        st.success(f"Loaded {len(df)} rows")
        st.dataframe(df.head())

        # Main action
        if st.button("Run analysis"):
            sentences = df[sentence_col].dropna().astype(str).tolist()
            if len(sentences) < 2:
                st.error("Need at least 2 sentences to run analysis")
            else:
                with st.spinner("Loading embedding model..."):
                    model = load_embedding_model(embedding_model_name)
                with st.spinner("Generating embeddings..."):
                    embeddings = compute_embeddings(model, sentences)
                st.write(f"Embeddings shape: {embeddings.shape}")

                # Find optimal k
                with st.spinner("Searching for optimal number of clusters..."):
                    k_opt, metrics_df = find_optimal_k(embeddings, max_k=int(max_k))
                st.write(f"**Chosen number of clusters (k)**: {k_opt}")
                st.dataframe(metrics_df)

                # Plot elbow + silhouette
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(metrics_df['k'], metrics_df['inertia'], 'o-')
                ax1.set_xlabel('k')
                ax1.set_ylabel('inertia')
                ax1.set_title('Elbow: inertia vs k')
                ax2.plot(metrics_df['k'], metrics_df['silhouette'], 'o-')
                ax2.set_xlabel('k')
                ax2.set_ylabel('silhouette')
                ax2.set_title('Silhouette vs k')
                st.pyplot(fig)

                # Run KMeans
                with st.spinner(f"Clustering into {k_opt} clusters..."):
                    km, labels = run_kmeans(embeddings, k_opt)
                    df['cluster'] = labels
                st.write(df['cluster'].value_counts().sort_index())
                st.write("Silhouette score:", round(silhouette_score(embeddings, labels), 4))

                # Visualize PCA and optional t-SNE
                with st.spinner("Creating visualizations (PCA / t-SNE)..."):
                    pca = PCA(n_components=2, random_state=42)
                    emb_pca = pca.fit_transform(embeddings)
                    fig1, ax = plt.subplots(figsize=(6, 5))
                    scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
                    ax.set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.1%})')
                    ax.set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.1%})')
                    ax.set_title('Clusters (PCA)')
                    st.pyplot(fig1)

                    if run_tsne and len(df) <= 5000:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(df)//4)))
                        emb_tsne = tsne.fit_transform(embeddings)
                        fig2, ax2 = plt.subplots(figsize=(6, 5))
                        ax2.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
                        ax2.set_title('Clusters (t-SNE)')
                        st.pyplot(fig2)

                # Topic modeling per cluster
                st.header("Per-cluster topic modeling & topic → sentence table")
                per_cluster_results = {}
                progress_bar = st.progress(0)
                total_clusters = len(df['cluster'].unique())
                for i, cid in enumerate(sorted(df['cluster'].unique())):
                    cluster_sentences = df[df['cluster'] == cid][sentence_col].tolist()
                    if len(cluster_sentences) < 3:
                        per_cluster_results[cid] = None
                    else:
                        with st.spinner(f"LDA on cluster {cid} ({len(cluster_sentences)} sentences)..."):
                            res = topic_model_for_cluster(cluster_sentences, num_topics=int(lda_topics_per_cluster), passes=int(lda_passes))
                            per_cluster_results[cid] = res
                            # show wordcloud for first topic if available
                            if res is not None:
                                lda = res['lda']
                                if lda.num_topics > 0:
                                    top_terms = dict(lda.show_topic(0, topn=20))
                                    wc = WordCloud(width=400, height=200, background_color='white')
                                    wc_img = wc.generate_from_frequencies(top_terms)
                                    figw, axw = plt.subplots(figsize=(6, 3))
                                    axw.imshow(wc_img, interpolation='bilinear')
                                    axw.axis('off')
                                    st.caption(f"Cluster {cid} — WordCloud for Topic 0")
                                    st.pyplot(figw)
                    progress_bar.progress((i+1)/total_clusters)

                # Build cluster-topic-sentence table
                result_table = build_cluster_topic_table(df, cluster_col='cluster', sentence_col=sentence_col, per_cluster_topics=per_cluster_results)

                # Limit examples per topic
                display_table = result_table.groupby(['cluster', 'topic']).head(int(samples_per_topic)).reset_index(drop=True)

                st.subheader("Topic table: cluster | topic name | example sentence")
                st.dataframe(display_table)

                # Allow downloading the full table
                def to_csv_bytes(df_):
                    return df_.to_csv(index=False).encode('utf-8')

                csv_bytes = to_csv_bytes(display_table)
                st.download_button("Download topic table (CSV)", data=csv_bytes, file_name='cluster_topic_table.csv', mime='text/csv')

                st.success("Analysis complete")

else:
    st.info("Upload a CSV or choose the example dataset from the sidebar to get started.")

# Footer/help
st.markdown("---")
st.markdown("**Notes & tips:**\n- If KneeLocator doesn't find a clear elbow, the app falls back to the k with the highest silhouette score.\n- For large datasets consider using a stronger machine or reducing data via PCA before KMeans.\n- t-SNE can be slow for >2000 rows; uncheck it if needed.")

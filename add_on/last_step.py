import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
import io

# Page configuration
st.set_page_config(page_title="Sentence Classification Tool", layout="wide")

st.title("📊 Sentence Classification Tool")
st.markdown("Upload your reference vectors and source data to classify sentences based on cosine similarity.")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# File uploads
st.sidebar.subheader("1. Upload Files")
reference_file = st.sidebar.file_uploader("Reference Vector File (tag, sentence)", type=['csv'])
source_file = st.sidebar.file_uploader("Source Data File (sentence)", type=['csv'])

# Model selection
st.sidebar.subheader("2. Select Embedding Model")
model_choice = st.sidebar.selectbox(
    "Embedding Model",
    ["sentence-transformers/all-MiniLM-L6-v2", "Google Gemini text-embedding-004"]
)

# API Key input for Gemini
api_key = None
if model_choice == "Google Gemini text-embedding-004":
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Threshold input
st.sidebar.subheader("3. Set Threshold")
threshold = st.sidebar.number_input(
    "Cosine Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    format="%.2f"
)

# Batch size for processing
BATCH_SIZE = 50

def load_and_validate_files(ref_file, src_file):
    """Load and validate the uploaded CSV files"""
    try:
        ref_df = pd.read_csv(ref_file)
        src_df = pd.read_csv(src_file)
        
        # Validate reference file columns
        if 'tag' not in ref_df.columns or 'sentence' not in ref_df.columns:
            st.error("Reference file must contain 'tag' and 'sentence' columns")
            return None, None
        
        # Validate source file columns
        if 'sentence' not in src_df.columns:
            st.error("Source file must contain 'sentence' column")
            return None, None
        
        # Remove any rows with missing sentences
        ref_df = ref_df.dropna(subset=['sentence', 'tag'])
        src_df = src_df.dropna(subset=['sentence'])
        
        return ref_df, src_df
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None

def get_embeddings_sentence_transformer(sentences, model):
    """Generate embeddings using sentence-transformers in batches"""
    all_embeddings = []
    
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        embeddings = model.encode(batch)
        all_embeddings.extend(embeddings)
    
    return np.array(all_embeddings)

def get_embeddings_gemini(sentences, client):
    """Generate embeddings using Google Gemini in batches"""
    all_embeddings = []
    
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=batch
        )
        for embedding in result.embeddings:
            all_embeddings.append(embedding.values)
    
    return np.array(all_embeddings)

def classify_sentences(ref_embeddings, src_embeddings, ref_df, src_df, threshold):
    """Classify source sentences based on cosine similarity with reference vectors"""
    predicted_tags = []
    similarity_scores = []
    matched_sentences = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_sentences = len(src_embeddings)
    
    for idx, src_emb in enumerate(src_embeddings):
        # Update progress
        progress = (idx + 1) / total_sentences
        progress_bar.progress(progress)
        status_text.text(f"Processing sentence {idx + 1} of {total_sentences}")
        
        # Calculate cosine similarity with all reference embeddings
        similarities = cosine_similarity([src_emb], ref_embeddings)[0]
        
        # Find the highest similarity score and its index
        max_idx = np.argmax(similarities)
        max_score = similarities[max_idx]
        
        # Check if score exceeds threshold
        if max_score >= threshold:
            predicted_tag = ref_df.iloc[max_idx]['tag']
            matched_sentence = ref_df.iloc[max_idx]['sentence']
        else:
            predicted_tag = "Other"
            matched_sentence = ""
        
        predicted_tags.append(predicted_tag)
        similarity_scores.append(round(max_score, 4))
        matched_sentences.append(matched_sentence)
    
    progress_bar.empty()
    status_text.empty()
    
    # Create a copy of the source dataframe and add new columns
    results_df = src_df.copy()
    results_df['Predicted Tag'] = predicted_tags
    results_df['Cosine Similarity Score'] = similarity_scores
    results_df['Matched Reference Sentence'] = matched_sentences
    
    return results_df


# Main processing
if reference_file and source_file:
    # Load and validate files
    ref_df, src_df = load_and_validate_files(reference_file, source_file)
    
    if ref_df is not None and src_df is not None:
        # Display file information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Reference Vectors")
            st.write(f"Total reference sentences: {len(ref_df)}")
            st.dataframe(ref_df.head(), use_container_width=True)
        
        with col2:
            st.subheader("📄 Source Data")
            st.write(f"Total source sentences: {len(src_df)}")
            st.dataframe(src_df.head(), use_container_width=True)
        
        # Process button
        if st.button("🚀 Start Classification", type="primary"):
            # Validate Gemini API key if needed
            if model_choice == "Google Gemini text-embedding-004" and not api_key:
                st.error("Please provide a Gemini API key")
            else:
                try:
                    with st.spinner("Loading embedding model..."):
                        if model_choice == "sentence-transformers/all-MiniLM-L6-v2":
                            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                            st.success("✅ Model loaded successfully")
                        else:
                            client = genai.Client(api_key=api_key)
                            st.success("✅ Gemini client initialized")
                    
                    # Generate embeddings for reference vectors
                    with st.spinner("Generating embeddings for reference vectors..."):
                        ref_sentences = ref_df['sentence'].tolist()
                        if model_choice == "sentence-transformers/all-MiniLM-L6-v2":
                            ref_embeddings = get_embeddings_sentence_transformer(ref_sentences, model)
                        else:
                            ref_embeddings = get_embeddings_gemini(ref_sentences, client)
                        st.success(f"✅ Generated {len(ref_embeddings)} reference embeddings")
                    
                    # Generate embeddings for source data
                    with st.spinner("Generating embeddings for source data..."):
                        src_sentences = src_df['sentence'].tolist()
                        if model_choice == "sentence-transformers/all-MiniLM-L6-v2":
                            src_embeddings = get_embeddings_sentence_transformer(src_sentences, model)
                        else:
                            src_embeddings = get_embeddings_gemini(src_sentences, client)
                        st.success(f"✅ Generated {len(src_embeddings)} source embeddings")
                    
                    # Classify sentences
                    st.subheader("🔄 Classification Progress")
                    results_df = classify_sentences(
                        ref_embeddings, 
                        src_embeddings, 
                        ref_df, 
                        src_df, 
                        threshold
                    )
                    
                    st.success("✅ Classification complete!")
                    
                    # Display results
                    st.subheader("📊 Results")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Sentences", len(results_df))
                    with col2:
                        classified = len(results_df[results_df['Predicted Tag'] != 'Other'])
                        st.metric("Classified", classified)
                    with col3:
                        other = len(results_df[results_df['Predicted Tag'] == 'Other'])
                        st.metric("Other", other)
                    
                    # Display results table
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_bytes = csv_buffer.getvalue().encode()
                    
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv_bytes,
                        file_name="classification_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

else:
    st.info("👈 Please upload both reference and source files to begin")
    
    # Instructions
    st.markdown("""
    ### 📖 Instructions
    
    1. **Upload Reference Vector File**: CSV with columns `tag` and `sentence`
    2. **Upload Source Data File**: CSV with column `sentence`
    3. **Select Embedding Model**: Choose between Sentence Transformers or Google Gemini
    4. **Set Threshold**: Adjust the cosine similarity threshold (0-1)
    5. **Click Start Classification**: Process your data
    
    The tool will classify each source sentence by finding the most similar reference sentence and assigning its tag if the similarity exceeds the threshold.
    """)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import sys
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qualivec.data import DataLoader
from qualivec.embedding import EmbeddingModel
from qualivec.matching import SemanticMatcher
from qualivec.classification import Classifier
from qualivec.evaluation import Evaluator
from qualivec.optimization import ThresholdOptimizer

# Set page config
st.set_page_config(
    page_title="QualiVec Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #048A81;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🔍 QualiVec Demo</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Qualitative Content Analysis with LLM Embeddings
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Data Upload", "🔧 Configuration", "🎯 Classification", "📈 Results"]
    )
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'reference_data' not in st.session_state:
        st.session_state.reference_data = None
    if 'labeled_data' not in st.session_state:
        st.session_state.labeled_data = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Upload":
        show_data_upload_page()
    elif page == "🔧 Configuration":
        show_configuration_page()
    elif page == "🎯 Classification":
        show_classification_page()
    elif page == "📈 Results":
        show_results_page()

def show_home_page():
    st.markdown('<div class="section-header">Welcome to QualiVec</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### What is QualiVec?
        
        QualiVec is a Python library that uses Large Language Model (LLM) embeddings for qualitative content analysis. It helps researchers and analysts classify text data by comparing it against reference examples.
        
        ### Key Features:
        - **Semantic Matching**: Uses advanced embedding models to find semantic similarity
        - **Threshold Optimization**: Automatically finds the best similarity threshold
        - **Comprehensive Evaluation**: Provides detailed metrics and visualizations
        - **Bootstrap Analysis**: Confidence intervals for robust evaluation
        
        ### How It Works:
        1. **Upload Data**: Provide reference examples and data to classify
        2. **Configure**: Set up embedding models and parameters
        3. **Optimize**: Find the best threshold for classification
        4. **Classify**: Apply the model to your data
        5. **Evaluate**: Get detailed performance metrics
        
        ### Getting Started:
        Use the sidebar to navigate through the demo. Start with **Data Upload** to begin your analysis.
        """)
    
    # Add sample data info
    st.markdown('<div class="section-header">Sample Data Format</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Reference Data Format:**")
        sample_ref = pd.DataFrame({
            'tag': ['Positive', 'Negative', 'Neutral'],
            'sentence': ['This is great!', 'This is terrible', 'This is okay']
        })
        st.dataframe(sample_ref, use_container_width=True)
        
    with col2:
        st.markdown("**Labeled Data Format:**")
        sample_labeled = pd.DataFrame({
            'sentence': ['I love this product', 'Not very good', 'Average quality'],
            'Label': ['Positive', 'Negative', 'Neutral']
        })
        st.dataframe(sample_labeled, use_container_width=True)

def show_data_upload_page():
    st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Reference Data")
        st.markdown("Upload a CSV file containing reference examples with columns: `tag` (class) and `sentence` (example text)")
        
        reference_file = st.file_uploader(
            "Choose reference data file",
            type=['csv'],
            key='reference_file'
        )
        
        if reference_file is not None:
            try:
                reference_df = pd.read_csv(reference_file)
                st.success("Reference data loaded successfully!")
                st.dataframe(reference_df.head(), use_container_width=True)
                
                # Validate columns
                required_cols = ['tag', 'sentence']
                missing_cols = [col for col in required_cols if col not in reference_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    # Prepare reference data
                    reference_df = reference_df.rename(columns={
                        'tag': 'class',
                        'sentence': 'matching_node'
                    })
                    st.session_state.reference_data = reference_df
                    
                    # Show statistics
                    st.markdown("**Data Statistics:**")
                    st.write(f"- Total examples: {len(reference_df)}")
                    st.write(f"- Unique classes: {reference_df['class'].nunique()}")
                    st.write(f"- Class distribution:")
                    st.write(reference_df['class'].value_counts())
                    
            except Exception as e:
                st.error(f"Error loading reference data: {str(e)}")
    
    with col2:
        st.markdown("### Labeled Data")
        st.markdown("Upload a CSV file containing data to classify with columns: `sentence` (text) and `Label` (true class)")
        
        labeled_file = st.file_uploader(
            "Choose labeled data file",
            type=['csv'],
            key='labeled_file'
        )
        
        if labeled_file is not None:
            try:
                labeled_df = pd.read_csv(labeled_file)
                st.success("Labeled data loaded successfully!")
                st.dataframe(labeled_df.head(), use_container_width=True)
                
                # Validate columns
                required_cols = ['sentence', 'Label']
                missing_cols = [col for col in required_cols if col not in labeled_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    # Prepare labeled data
                    labeled_df = labeled_df.rename(columns={'Label': 'label'})
                    labeled_df['label'] = labeled_df['label'].replace('0', 'Other')
                    st.session_state.labeled_data = labeled_df
                    
                    # Show statistics
                    st.markdown("**Data Statistics:**")
                    st.write(f"- Total samples: {len(labeled_df)}")
                    st.write(f"- Unique labels: {labeled_df['label'].nunique()}")
                    st.write(f"- Label distribution:")
                    st.write(labeled_df['label'].value_counts())
                    
            except Exception as e:
                st.error(f"Error loading labeled data: {str(e)}")
    
    # Show data compatibility check
    if st.session_state.reference_data is not None and st.session_state.labeled_data is not None:
        st.markdown('<div class="section-header">Data Compatibility Check</div>', unsafe_allow_html=True)
        
        ref_classes = set(st.session_state.reference_data['class'].unique())
        labeled_classes = set(st.session_state.labeled_data['label'].unique())
        
        # Check for unknown classes
        unknown_classes = labeled_classes - ref_classes
        
        if unknown_classes:
            st.warning(f"Warning: Labels in labeled data not found in reference data: {unknown_classes}")
        else:
            st.success("✅ Data compatibility check passed!")
        
        # Show class overlap
        st.markdown("**Class Overlap Analysis:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Reference Classes", len(ref_classes))
        with col2:
            st.metric("Labeled Classes", len(labeled_classes))
        with col3:
            st.metric("Common Classes", len(ref_classes.intersection(labeled_classes)))

def show_configuration_page():
    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
    
    # Check if data is loaded
    if st.session_state.reference_data is None or st.session_state.labeled_data is None:
        st.warning("Please upload both reference and labeled data first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Embedding Model")
        
        # Model type selection
        model_type = st.selectbox(
            "Choose model type",
            ["HuggingFace", "Gemini"],
            help="Select the type of embedding model to use"
        )
        
        # Model selection based on type
        if model_type == "HuggingFace":
            model_options = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/distilbert-base-nli-mean-tokens"
            ]
            
            selected_model = st.selectbox(
                "Choose HuggingFace model",
                model_options,
                help="Select the pre-trained HuggingFace model for generating embeddings"
            )
        else:  # Gemini
            gemini_models = [
                "gemini-embedding-001",
                "text-embedding-004"
            ]
            
            selected_model = st.selectbox(
                "Choose Gemini model",
                gemini_models,
                help="Select the Gemini embedding model for generating embeddings"
            )
            
            # Calculate total texts to process
            total_texts = 0
            if st.session_state.reference_data is not None:
                total_texts += len(st.session_state.reference_data)
            if st.session_state.labeled_data is not None:
                total_texts += len(st.session_state.labeled_data)
            
            st.warning(
                f"⚠️ **Gemini API Rate Limits (Free Tier)**\\n\\n"
                f"- 1,500 requests per day\\n"
                f"- Each batch of 100 texts = 1 request\\n"
                f"- Your current dataset: ~{total_texts} texts\\n"
                f"- Estimated requests needed: ~{(total_texts // 100) + 1}\\n\\n"
                f"If you exceed quota, consider:\\n"
                f"1. Using a smaller dataset\\n"
                f"2. Switching to HuggingFace models (no limits)\\n"
                f"3. Upgrading to a paid API plan"
            )
            
            st.info("💡 Note: Using Gemini embeddings requires GOOGLE_API_KEY environment variable to be set.")
        
        st.markdown("### Initial Threshold")
        initial_threshold = st.slider(
            "Initial similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Cosine similarity threshold for classification"
        )
    
    with col2:
        st.markdown("### Optimization Parameters")
        
        optimize_threshold = st.checkbox(
            "Enable threshold optimization",
            value=True,
            help="Automatically find the best threshold"
        )
        
        if optimize_threshold:
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                start_threshold = st.slider(
                    "Start threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05
                )
                
                end_threshold = st.slider(
                    "End threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.05
                )
            
            with col2_2:
                step_size = st.slider(
                    "Step size",
                    min_value=0.005,
                    max_value=0.05,
                    value=0.01,
                    step=0.005
                )
                
                optimization_metric = st.selectbox(
                    "Optimization metric",
                    ["f1_macro", "accuracy", "precision_macro", "recall_macro"]
                )
    
    # Load models button
    if st.button("Initialize Models", type="primary"):
        with st.spinner("Loading models... This may take a few minutes."):
            try:
                # Initialize classifier
                classifier = Classifier(verbose=False)
                
                # Determine model type parameter
                model_type_param = "gemini" if model_type == "Gemini" else "huggingface"
                
                classifier.load_models(
                    model_name=selected_model,
                    model_type=model_type_param,
                    threshold=initial_threshold
                )
                
                # Prepare reference vectors
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_ref:
                    tmp_ref_path = tmp_ref.name
                    st.session_state.reference_data.to_csv(tmp_ref_path, index=False)
                
                try:
                    reference_data = classifier.prepare_reference_vectors(
                        reference_path=tmp_ref_path,
                        class_column='class',
                        node_column='matching_node'
                    )
                finally:
                    # Ensure file is deleted even if an error occurs
                    try:
                        os.unlink(tmp_ref_path)
                    except (OSError, PermissionError):
                        pass  # File might already be deleted or locked
                
                st.session_state.classifier = classifier
                st.session_state.reference_vectors = reference_data
                st.session_state.config = {
                    'model_type': model_type,
                    'model_name': selected_model,
                    'initial_threshold': initial_threshold,
                    'optimize_threshold': optimize_threshold,
                    'start_threshold': start_threshold if optimize_threshold else None,
                    'end_threshold': end_threshold if optimize_threshold else None,
                    'step_size': step_size if optimize_threshold else None,
                    'optimization_metric': optimization_metric if optimize_threshold else None
                }
                
                st.success("✅ Models initialized successfully!")
                
            except Exception as e:
                st.error(f"Error initializing models: {str(e)}")
    
    # Show current configuration
    if st.session_state.classifier is not None:
        st.markdown('<div class="section-header">Current Configuration</div>', unsafe_allow_html=True)
        
        config = st.session_state.config
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Model Settings:**")
            st.write(f"- Model type: {config['model_type']}")
            st.write(f"- Model: {config['model_name']}")
            st.write(f"- Initial threshold: {config['initial_threshold']}")
        
        with col2:
            st.markdown("**Optimization:**")
            st.write(f"- Enabled: {config['optimize_threshold']}")
            if config['optimize_threshold']:
                st.write(f"- Range: {config['start_threshold']:.2f} - {config['end_threshold']:.2f}")
                st.write(f"- Step: {config['step_size']:.3f}")
        
        with col3:
            st.markdown("**Data:**")
            st.write(f"- Reference examples: {len(st.session_state.reference_data)}")
            st.write(f"- Labeled samples: {len(st.session_state.labeled_data)}")

def show_classification_page():
    st.markdown('<div class="section-header">Classification & Optimization</div>', unsafe_allow_html=True)
    
    # Check if models are loaded
    if st.session_state.classifier is None:
        st.warning("Please configure and initialize models first.")
        return
    
    # Run classification
    if st.button("Run Classification", type="primary"):
        # New: progress bar and status placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.info("Starting classification...")
        
        with st.spinner("Running classification and optimization..."):
            try:
                # Save labeled data to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_labeled:
                    tmp_labeled_path = tmp_labeled.name
                    st.session_state.labeled_data.to_csv(tmp_labeled_path, index=False)
                
                try:
                    # Run optimization if enabled
                    if st.session_state.config['optimize_threshold']:
                        status_text.info("Running threshold optimization...")
                        progress_bar.progress(10)
                        
                        optimization_results = st.session_state.classifier.evaluate_classification(
                            labeled_path=tmp_labeled_path,
                            reference_data=st.session_state.reference_vectors,
                            sentence_column='sentence',
                            label_column='label',
                            optimize_threshold=True,
                            start=st.session_state.config['start_threshold'],
                            end=st.session_state.config['end_threshold'],
                            step=st.session_state.config['step_size']
                        )
                        
                        st.session_state.optimization_results = optimization_results
                        optimal_threshold = optimization_results["optimal_threshold"]
                        
                        # Update classifier with optimal threshold
                        st.session_state.classifier.matcher = SemanticMatcher(
                            threshold=optimal_threshold, 
                            verbose=False
                        )
                        
                        progress_bar.progress(40)
                        status_text.success(f"Optimization completed. Optimal threshold: {optimal_threshold:.4f}")
                        
                    else:
                        optimal_threshold = st.session_state.config['initial_threshold']
                        progress_bar.progress(20)
                        status_text.info(f"Using initial threshold: {optimal_threshold:.4f}")
					
					# Run evaluation
					status_text.info("Generating embeddings...")
					progress_bar.progress(50)
					
					embedding_model = st.session_state.classifier.embedding_model
					data_loader = DataLoader(verbose=False)
					full_df = data_loader.load_labeled_data(tmp_labeled_path, label_column='label')
					
					# Generate embeddings
					full_embeddings = embedding_model.embed_dataframe(full_df, text_column='sentence')
					
					progress_bar.progress(70)
					status_text.info("Classifying with semantic matcher...")
					
					# Classify
					match_results = st.session_state.classifier.matcher.match(
						full_embeddings, 
						st.session_state.reference_vectors
					)
					predicted_labels = match_results["predicted_class"].tolist()
					true_labels = full_df['label'].tolist()
					
					progress_bar.progress(80)
					status_text.info("Evaluating predicted labels...")
					
					# Evaluate
					evaluator = Evaluator(verbose=False)
					eval_results = evaluator.evaluate(
						true_labels=true_labels,
						predicted_labels=predicted_labels,
						class_names=list(set(true_labels) | set(predicted_labels))
					)
					
					progress_bar.progress(90)
					status_text.info("Running bootstrap evaluation...")
					
					# Bootstrap evaluation
					bootstrap_results = evaluator.bootstrap_evaluate(
						true_labels=true_labels,
						predicted_labels=predicted_labels,
						n_iterations=100
					)
					
					progress_bar.progress(98)
					
					st.session_state.evaluation_results = eval_results
					st.session_state.bootstrap_results = bootstrap_results
					st.session_state.predictions = {
						'true_labels': true_labels,
						'predicted_labels': predicted_labels,
						'match_results': match_results,
						'full_df': full_df
					}
					
				finally:
					# Ensure temporary file is deleted
					try:
						os.unlink(tmp_labeled_path)
					except (OSError, PermissionError):
						pass  # File might already be deleted or locked
					
					progress_bar.progress(100)
					status_text.success("Classification completed successfully!")
					st.success("✅ Classification completed successfully!")
					
            except Exception as e:
                progress_bar.empty()
                status_text.error(f"Error during classification: {str(e)}")
                st.error(f"Error during classification: {str(e)}")
	
	# Show optimization results if available
    if st.session_state.optimization_results is not None:
        st.markdown('<div class="section-header">Optimization Results</div>', unsafe_allow_html=True)
        
        results = st.session_state.optimization_results
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Optimal Threshold",
                f"{results['optimal_threshold']:.4f}"
            )
        
        with col2:
            st.metric(
                "Accuracy",
                f"{results['optimal_metrics']['accuracy']:.4f}"
            )
        
        with col3:
            st.metric(
                "F1 Score",
                f"{results['optimal_metrics']['f1_macro']:.4f}"
            )
        
        with col4:
            st.metric(
                "Precision",
                f"{results['optimal_metrics']['precision_macro']:.4f}"
            )
        
        # Plot optimization curve
        st.markdown("### Optimization Curve")
        
        opt_results = results["results_by_threshold"]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall'),
            vertical_spacing=0.1
        )
        
        thresholds = opt_results["thresholds"]
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=thresholds, y=opt_results["accuracy"], name="Accuracy"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=thresholds, y=opt_results["f1_macro"], name="F1 Score"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=thresholds, y=opt_results["precision_macro"], name="Precision"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=thresholds, y=opt_results["recall_macro"], name="Recall"),
            row=2, col=2
        )
        
        # Add optimal threshold line to each subplot using shapes
        optimal_thresh = results['optimal_threshold']
        
        # Add vertical line as shapes to each subplot
        shapes = []
        for row in range(1, 3):
            for col in range(1, 3):
                # Calculate the subplot domain
                xaxis = f'x{(row-1)*2 + col}' if (row-1)*2 + col > 1 else 'x'
                shapes.append(
                    dict(
                        type="line",
                        x0=optimal_thresh, x1=optimal_thresh,
                        y0=0, y1=1,
                        yref=f"y{(row-1)*2 + col} domain" if (row-1)*2 + col > 1 else "y domain",
                        xref=xaxis,
                        line=dict(color="red", width=2, dash="dash")
                    )
                )
        
        fig.update_layout(shapes=shapes)
        
        fig.update_layout(
            title="Threshold Optimization Results",
            showlegend=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_results_page():
    st.markdown('<div class="section-header">Results & Evaluation</div>', unsafe_allow_html=True)
    
    # Check if evaluation results are available
    if st.session_state.evaluation_results is None:
        st.warning("Please run classification first to see results.")
        return
    
    eval_results = st.session_state.evaluation_results
    
    # Performance metrics
    st.markdown("### Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Accuracy",
            f"{eval_results['accuracy']:.4f}"
        )
    
    with col2:
        st.metric(
            "Macro F1 Score",
            f"{eval_results['f1_macro']:.4f}"
        )
    
    with col3:
        st.metric(
            "Macro Precision",
            f"{eval_results['precision_macro']:.4f}"
        )
    
    with col4:
        st.metric(
            "Macro Recall",
            f"{eval_results['recall_macro']:.4f}"
        )
    
    # Class-wise metrics
    st.markdown("### Class-wise Performance")
    
    class_metrics_df = pd.DataFrame({
        'Class': list(eval_results['class_metrics']['precision'].keys()),
        'Precision': list(eval_results['class_metrics']['precision'].values()),
        'Recall': list(eval_results['class_metrics']['recall'].values()),
        'F1-Score': list(eval_results['class_metrics']['f1'].values()),
        'Support': list(eval_results['class_metrics']['support'].values())
    })
    
    st.dataframe(class_metrics_df, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    
    cm = eval_results['confusion_matrix']
    class_names = eval_results['confusion_matrix_labels']
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        text_auto=True,
        title="Confusion Matrix"
    )
    
    fig.update_layout(
        width=600,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bootstrap Results
    if st.session_state.bootstrap_results is not None:
        st.markdown("### Bootstrap Confidence Intervals")
        
        bootstrap_results = st.session_state.bootstrap_results
        
        # Debug: show available keys
        if 'confidence_intervals' in bootstrap_results:
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            
            for metric in metrics:
                if metric in bootstrap_results['confidence_intervals']:
                    ci_data = bootstrap_results['confidence_intervals'][metric]
                    st.markdown(f"**{metric.replace('_', ' ').title()}:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Check available confidence levels
                    available_levels = list(ci_data.keys())
                    
                    with col1:
                        if '0.95' in ci_data:
                            ci_95 = ci_data['0.95']
                            if isinstance(ci_95, dict):
                                st.write(f"95% CI: [{ci_95['lower']:.4f}, {ci_95['upper']:.4f}]")
                            elif isinstance(ci_95, (list, tuple)) and len(ci_95) >= 2:
                                st.write(f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
                            else:
                                st.write("95% CI: Format not recognized")
                        elif 0.95 in ci_data:
                            ci_95 = ci_data[0.95]
                            if isinstance(ci_95, dict):
                                st.write(f"95% CI: [{ci_95['lower']:.4f}, {ci_95['upper']:.4f}]")
                            elif isinstance(ci_95, (list, tuple)) and len(ci_95) >= 2:
                                st.write(f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
                            else:
                                st.write("95% CI: Format not recognized")
                        else:
                            st.write("95% CI: Not available")
                    
                    with col2:
                        if '0.99' in ci_data:
                            ci_99 = ci_data['0.99']
                            if isinstance(ci_99, dict):
                                st.write(f"99% CI: [{ci_99['lower']:.4f}, {ci_99['upper']:.4f}]")
                            elif isinstance(ci_99, (list, tuple)) and len(ci_99) >= 2:
                                st.write(f"99% CI: [{ci_99[0]:.4f}, {ci_99[1]:.4f}]")
                            else:
                                st.write("99% CI: Format not recognized")
                        elif 0.99 in ci_data:
                            ci_99 = ci_data[0.99]
                            if isinstance(ci_99, dict):
                                st.write(f"99% CI: [{ci_99['lower']:.4f}, {ci_99['upper']:.4f}]")
                            elif isinstance(ci_99, (list, tuple)) and len(ci_99) >= 2:
                                st.write(f"99% CI: [{ci_99[0]:.4f}, {ci_99[1]:.4f}]")
                            else:
                                st.write("99% CI: Format not recognized")
                        else:
                            st.write("99% CI: Not available")
                    
                    with col3:
                        if 'point_estimates' in bootstrap_results and metric in bootstrap_results['point_estimates']:
                            st.write(f"Point Estimate: {bootstrap_results['point_estimates'][metric]:.4f}")
                        else:
                            st.write("Point Estimate: Not available")
        else:
            st.info("Bootstrap confidence intervals not available.")
        
        # Bootstrap Distribution Plot
        st.markdown("### Bootstrap Distributions")
        
        if 'bootstrap_distribution' in bootstrap_results:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall')
            )
            
            distributions = bootstrap_results['bootstrap_distribution']
            
            if 'accuracy' in distributions:
                fig.add_trace(
                    go.Histogram(x=distributions['accuracy'], name="Accuracy", nbinsx=30),
                    row=1, col=1
                )
            if 'f1_macro' in distributions:
                fig.add_trace(
                    go.Histogram(x=distributions['f1_macro'], name="F1 Score", nbinsx=30),
                    row=1, col=2
                )
            if 'precision_macro' in distributions:
                fig.add_trace(
                    go.Histogram(x=distributions['precision_macro'], name="Precision", nbinsx=30),
                    row=2, col=1
                )
            if 'recall_macro' in distributions:
                fig.add_trace(
                    go.Histogram(x=distributions['recall_macro'], name="Recall", nbinsx=30),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Bootstrap Distributions",
                showlegend=False,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Bootstrap distributions not available.")
    
    # Sample predictions
    if 'predictions' in st.session_state:
        st.markdown("### Sample Predictions")
        
        predictions = st.session_state.predictions
        sample_df = predictions['full_df'].copy()
        sample_df['predicted_class'] = predictions['predicted_labels']
        sample_df['true_class'] = predictions['true_labels']
        sample_df['similarity_score'] = predictions['match_results']['similarity_score']
        sample_df['correct'] = sample_df['predicted_class'] == sample_df['true_class']
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            show_correct = st.checkbox("Show correct predictions", value=True)
        
        with col2:
            show_incorrect = st.checkbox("Show incorrect predictions", value=True)
        
        # Filter data
        if show_correct and show_incorrect:
            filtered_df = sample_df
        elif show_correct:
            filtered_df = sample_df[sample_df['correct'] == True]
        elif show_incorrect:
            filtered_df = sample_df[sample_df['correct'] == False]
        else:
            filtered_df = pd.DataFrame()
        
        if not filtered_df.empty:
            # Sample random rows
            n_samples = min(20, len(filtered_df))
            sample_rows = filtered_df.sample(n=n_samples) if len(filtered_df) > n_samples else filtered_df
            
            display_df = sample_rows[['sentence', 'true_class', 'predicted_class', 'similarity_score', 'correct']].reset_index(drop=True)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No predictions to show with current filters.")
    
    # Download results
    st.markdown("### Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download class-wise metrics
        csv_metrics = class_metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Class Metrics",
            data=csv_metrics,
            file_name="class_metrics.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download predictions
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions
            results_df = predictions['full_df'].copy()
            results_df['predicted_class'] = predictions['predicted_labels']
            results_df['similarity_score'] = predictions['match_results']['similarity_score']
            
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv_results,
                file_name="predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

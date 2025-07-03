# QualiVec Streamlit Demo

This Streamlit application provides an interactive demonstration of the QualiVec library for qualitative content analysis using LLM embeddings.

## Features

- **Interactive Data Upload**: Upload your own CSV files for reference and labeled data
- **Model Configuration**: Choose from different pre-trained embedding models
- **Threshold Optimization**: Automatically find the optimal similarity threshold
- **Real-time Classification**: See classification results as they happen
- **Comprehensive Evaluation**: View detailed performance metrics and visualizations
- **Bootstrap Analysis**: Get confidence intervals for robust evaluation

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Run the App**:
   ```bash
   cd app
   uv run run_demo.py
   ```

3. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`

## Data Format Requirements

### Reference Data (CSV)
Your reference data should contain:
- `tag`: The class/category label
- `sentence`: The example text for that category

Example:
```csv
tag,sentence
Positive,This is absolutely fantastic!
Negative,This is terrible and disappointing
Neutral,This is okay I guess
```

### Labeled Data (CSV)
Your labeled data should contain:
- `sentence`: The text to be classified
- `Label`: The true class/category (for evaluation)

Example:
```csv
sentence,Label
I love this product so much!,Positive
Not very good quality,Negative
Average product nothing special,Neutral
```

## Navigation

The app is organized into 5 main sections:

1. **🏠 Home**: Overview and introduction to QualiVec
2. **📊 Data Upload**: Upload your reference and labeled data files
3. **🔧 Configuration**: Set up embedding models and parameters
4. **🎯 Classification**: Run the classification and optimization process
5. **📈 Results**: View detailed results and download outputs
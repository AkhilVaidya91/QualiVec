"""QualiVec: Qualitative Content Analysis with LLM Embeddings."""

from qualivec.data import DataLoader
from qualivec.sampling import Sampler
from qualivec.embedding import EmbeddingModel
from qualivec.matching import SemanticMatcher
from qualivec.evaluation import Evaluator
from qualivec.optimization import ThresholdOptimizer
from qualivec.classification import Classifier

__version__ = "0.1.0"

def main() -> None:
    print("QualiVec: Qualitative Content Analysis with LLM Embeddings")
    print(f"Version: {__version__}")

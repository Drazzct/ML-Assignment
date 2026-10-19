"""
Module: Traditional Feature Extraction
BoW, TF-IDF, and character n-gram feature extraction.

Produces sparse feature matrices and saves them via `feature_io`. Sparse
matrices can be passed directly to scikit-learn estimators.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from modules.config import TraditionalFeatureConfig
from modules.feature_io import save_features


# ──────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────

def _build_vectorizer(config: TraditionalFeatureConfig):
    """
    Returns a scikit-learn vectorizer based on the config.

    Implementation Guide:
    1. If `config.feature_type == 'bow'`:
       return `CountVectorizer(max_features=config.max_features, ngram_range=config.ngram_range)`.
    2. If `'tfidf'`:
       return `TfidfVectorizer(max_features=config.max_features, ngram_range=config.ngram_range)`.
    3. If `'ngram'`:
       return `TfidfVectorizer(max_features=config.max_features, ngram_range=config.ngram_range, analyzer='char_wb')`.
       (Character n-grams capture sub-word patterns.)
    4. Else raise `ValueError`.
    """
    if config.feature_type == "bow":
        return CountVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
        )
    if config.feature_type == "tfidf":
        return TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
        )
    if config.feature_type == "ngram":
        return TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            analyzer="char_wb",
        )
    raise ValueError(
        f"Unsupported feature_type {config.feature_type!r}; "
        "expected 'bow', 'tfidf', or 'ngram'."
    )


# ──────────────────────────────────────────────
#  Extractor
# ──────────────────────────────────────────────

class TraditionalFeatureExtractor:
    def __init__(self, config: TraditionalFeatureConfig = None):
        """
        Initializes the vectorizer from a config.

        Implementation Guide:
        1. Default to `TraditionalFeatureConfig()` if `config` is None.
        2. `self.vectorizer = _build_vectorizer(config)`.
        """
        self.config = config or TraditionalFeatureConfig()
        self.vectorizer = _build_vectorizer(self.config)

    def extract(self, texts, is_train=True):
        """
        Fits (if train) and transforms texts into a sparse feature matrix.

        Implementation Guide:
        1. If `is_train`: `sparse_matrix = self.vectorizer.fit_transform(texts)`.
        2. Else: `sparse_matrix = self.vectorizer.transform(texts)`.

        Args:
            texts (list[str] | pd.Series): Input texts.
            is_train (bool): Whether to fit the vectorizer.
        Returns:
            scipy.sparse.csr_matrix: Sparse feature matrix of shape (N, D).
        """
        if is_train:
            return self.vectorizer.fit_transform(texts)
        return self.vectorizer.transform(texts)

    def extract_and_save(self, texts, filename, is_train=True):
        """
        Extracts features and saves them to disk in one step.

        Implementation Guide:
        1. `embeddings = self.extract(texts, is_train)`.
        2. `save_features(embeddings, filename, self.config.features_dir, self.config.save_format)`.
        3. Return `embeddings`.

        Args:
            texts: Input texts.
            filename (str): Base filename (e.g. 'tfidf_train').
            is_train (bool): Whether to fit the vectorizer.
        Returns:
            scipy.sparse.csr_matrix: Sparse feature matrix.
        """
        features = self.extract(texts, is_train=is_train)
        save_features(
            features,
            filename,
            self.config.features_dir,
            self.config.save_format,
        )
        return features

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception as import_error:  # pragma: no cover - optional at runtime
    SentenceTransformer = None  # type: ignore[assignment]
    st_util = None  # type: ignore[assignment]


class SemanticTagger:
    """Embedding-based multi-label tagger using sentence-transformers.

    This class computes cosine similarity between an input text embedding and
    per-label prototype embeddings derived from seed texts.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not available. Please install it to use SemanticTagger."
            )
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=device)
        self._label_to_embedding: Dict[str, np.ndarray] = {}

    def set_label_prototypes(self, label_to_seed_text: Dict[str, str]) -> None:
        """Set prototype embeddings for each label based on seed text.

        The seed text should capture the concept of the label (e.g., label name
        plus synonyms/keywords).
        """
        labels: List[str] = list(label_to_seed_text.keys())
        seeds: List[str] = [label_to_seed_text[label] for label in labels]
        embeddings = self._encode(seeds)
        self._label_to_embedding = {
            label: embedding for label, embedding in zip(labels, embeddings)
        }

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def tag_text(
        self,
        text: str,
        *,
        threshold: float = 0.3,
        max_labels: int = 5,
    ) -> List[str]:
        if not text or not self._label_to_embedding:
            return []

        text_embedding = self._encode([text])[0]
        labels_and_scores: List[Tuple[str, float]] = []
        for label, proto_embedding in self._label_to_embedding.items():
            # Since both vectors are normalized, cosine similarity == dot product
            score = float(np.dot(text_embedding, proto_embedding))
            labels_and_scores.append((label, score))

        # Sort by descending score
        labels_and_scores.sort(key=lambda pair: pair[1], reverse=True)

        # Filter by threshold and cap by max_labels
        selected = [label for label, score in labels_and_scores if score >= threshold]
        if max_labels > 0:
            selected = selected[:max_labels]
        return selected


_DEFAULT_TAGGER: Optional[SemanticTagger] = None
_DEFAULT_LABEL_SIGNATURE: Optional[Tuple[str, ...]] = None


def get_or_create_default_tagger(
    label_to_keywords: Dict[str, List[str]],
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> SemanticTagger:
    """Get a cached default tagger configured with provided labels/keywords.

    The cache key is the tuple of label names. If the set of labels changes,
    the tagger is rebuilt.
    """
    global _DEFAULT_TAGGER, _DEFAULT_LABEL_SIGNATURE

    label_names_signature = tuple(sorted(label_to_keywords.keys()))
    if _DEFAULT_TAGGER is not None and _DEFAULT_LABEL_SIGNATURE == label_names_signature:
        return _DEFAULT_TAGGER

    chosen_model = model_name or os.getenv(
        "SCA_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    tagger = SemanticTagger(model_name=chosen_model, device=device)

    # Build seed text for each label by combining the label name and keywords
    label_to_seed_text: Dict[str, str] = {}
    for label, keywords in label_to_keywords.items():
        pretty_label = label.replace("-", " ")
        # Prioritize label name and include representative keywords/stems
        seed_text = f"{pretty_label}. Keywords: " + ", ".join(sorted(set(keywords)))
        label_to_seed_text[label] = seed_text

    tagger.set_label_prototypes(label_to_seed_text)

    _DEFAULT_TAGGER = tagger
    _DEFAULT_LABEL_SIGNATURE = label_names_signature
    return tagger


def semantic_tags_for_text(
    text: str,
    label_to_keywords: Dict[str, List[str]],
    *,
    threshold: Optional[float] = None,
    max_labels: Optional[int] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> List[str]:
    """Convenience function to compute semantic tags for a text.

    Uses environment variables for defaults if threshold/max_labels not provided:
      - SCA_SEMANTIC_THRESHOLD (float)
      - SCA_SEMANTIC_MAX_LABELS (int)
    """
    # Allow both package and module import contexts by avoiding relative imports here
    tagger = get_or_create_default_tagger(label_to_keywords, model_name=model_name, device=device)
    chosen_threshold = (
        threshold if threshold is not None else float(os.getenv("SCA_SEMANTIC_THRESHOLD", "0.3"))
    )
    chosen_max_labels = (
        max_labels if max_labels is not None else int(os.getenv("SCA_SEMANTIC_MAX_LABELS", "5"))
    )
    return tagger.tag_text(text or "", threshold=chosen_threshold, max_labels=chosen_max_labels)



"""
Deterministic, offline sentiment analysis module for normalized social post data.

This module provides rule-based sentiment and subjectivity scoring specifically
tailored to cryptocurrency and financial social media posts (e.g. X/Twitter).
It operates completely offline without external API dependencies or heavy pre-trained
weights, guaranteeing deterministic reproducibility and high processing throughput.

Scoring Rules & Methodological Architecture:
    1. Text Preprocessing:
       - Input normalization to lowercase for case-insensitive matching.
       - Preservation of punctuation and emoji signals.
       - Regex-based tokenization supporting contractions, hashtags, and symbols.
    2. Multi-Word Phrase Matching:
       - Pre-evaluation of idiomatic crypto expressions (e.g. 'to the moon',
         'all time high', 'bear market', 'dead cat bounce', 'rug pull').
    3. Crypto-Domain & General Sentiment Lexicon:
       - Domain-specific terms with calibrated valence scores in [-1.0, 1.0]
         (e.g., 'bullish': +0.8, 'breakout': +0.7, 'crash': -0.9, 'rekt': -0.8).
       - General sentiment terms (e.g., 'great': +0.6, 'terrible': -0.7).
    4. Negation Handling:
       - Triggers: 'not', 'no', 'never', "n't", 'without', 'hardly', 'scarcely', etc.
       - Local sliding window of up to 3 tokens following a negation flips the sign
         and applies a damping coefficient (0.8x) to represent dampened inverted sentiment.
       - Exceptions: Idiomatic constructions like 'not only' and 'not just' do not
         activate the negation window for subsequent terms.
    5. Intensifiers & Diminishers:
       - Modifiers immediately preceding a sentiment token scale the valence
         (e.g., 'extremely': 1.3x, 'slightly': 0.7x).
    6. Non-Linear Bounding:
       - Raw accumulated valence is transformed via hyperbolic tangent:
         sentiment_score = tanh(raw_valence / 2.0), strictly bounded to [-1.0, 1.0].
       - If raw valence is 0.0, the score is exactly 0.0.
    7. Subjectivity Heuristic:
       - Defined as the proportion of subjective tokens (sentiment words, intensifiers,
         and opinion markers like 'think', 'believe', 'feel', 'imo') relative to
         the total word count, bounded to [0.0, 1.0].
       - Note: This heuristic measures lexical density of opinion markers and is NOT
         a validated psychological, cognitive, or clinical measure.
    8. Safe Fallback:
       - None, NaN, empty strings, or whitespace-only inputs safely evaluate to
         sentiment_score = 0.0, subjectivity_score = 0.0, and label = 'neutral'.

Known Limitations of Rule-Based Sentiment:
    - Sarcasm & Irony: Lexicon-based methods cannot reliably capture satirical or
      ironic phrasing (e.g., 'Oh fantastic, another 25% liquidation cascade').
    - Financial Polysemy & Context: Words like 'mine', 'gas', or 'fork' have dual
      meanings depending on technical versus literal context.
    - Syntactic Complexity: Long-range dependent clauses and double negatives beyond
      the local sliding window may be misclassified.
    - Lexical Drift: Rapidly emerging slang, abbreviations, or neologisms on crypto
      social platforms require ongoing lexicon maintenance.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import math
import re
import pandas as pd

# Multi-word idiomatic phrases and their valence weights [-1.0, 1.0]
MULTI_WORD_PHRASES: Dict[str, float] = {
    "to the moon": 0.9,
    "all time high": 0.8,
    "ath": 0.8,
    "bull market": 0.8,
    "higher high": 0.6,
    "higher highs": 0.6,
    "short squeeze": 0.7,
    "strong support": 0.6,
    "break out": 0.7,
    "buy the dip": 0.7,
    "buying the dip": 0.7,
    "bear market": -0.8,
    "dead cat bounce": -0.7,
    "lower low": -0.6,
    "lower lows": -0.6,
    "sell off": -0.7,
    "selling off": -0.7,
    "rug pull": -0.9,
    "panic sell": -0.8,
    "panic selling": -0.8,
}

# Domain-specific and general unigram sentiment lexicon [-1.0, 1.0]
SENTIMENT_LEXICON: Dict[str, float] = {
    # Crypto Bullish / Positive
    "bullish": 0.85,
    "bull": 0.6,
    "rally": 0.75,
    "rallies": 0.75,
    "rallying": 0.75,
    "pump": 0.7,
    "pumping": 0.7,
    "breakout": 0.75,
    "moon": 0.8,
    "accumulate": 0.6,
    "accumulation": 0.65,
    "accumulating": 0.6,
    "buy": 0.5,
    "buying": 0.55,
    "buyer": 0.5,
    "buyers": 0.5,
    "surge": 0.7,
    "surging": 0.7,
    "gain": 0.6,
    "gains": 0.65,
    "green": 0.45,
    "hodl": 0.6,
    "adoption": 0.55,
    "profit": 0.65,
    "profits": 0.65,
    "profitable": 0.65,
    "rebound": 0.6,
    "rebounds": 0.6,
    "recovery": 0.6,
    "soar": 0.75,
    "soaring": 0.75,
    "gem": 0.7,
    "undervalued": 0.6,
    "upside": 0.6,
    "support": 0.4,
    "holding": 0.35,

    # General Positive
    "good": 0.5,
    "great": 0.7,
    "excellent": 0.8,
    "strong": 0.65,
    "best": 0.75,
    "promising": 0.6,
    "growth": 0.55,
    "positive": 0.6,
    "win": 0.6,
    "winning": 0.65,
    "solid": 0.55,
    "healthy": 0.55,
    "super": 0.6,

    # Crypto Bearish / Negative
    "bearish": -0.85,
    "bear": -0.6,
    "crash": -0.9,
    "crashed": -0.9,
    "crashing": -0.9,
    "dump": -0.75,
    "dumping": -0.75,
    "dumped": -0.75,
    "rug": -0.85,
    "rugged": -0.85,
    "rekt": -0.85,
    "liquidate": -0.75,
    "liquidated": -0.8,
    "liquidation": -0.8,
    "liquidations": -0.8,
    "capitulate": -0.8,
    "capitulation": -0.85,
    "fud": -0.65,
    "scam": -0.9,
    "scammer": -0.9,
    "scams": -0.9,
    "bleed": -0.7,
    "bleeding": -0.75,
    "drop": -0.5,
    "dropped": -0.55,
    "dropping": -0.55,
    "loss": -0.65,
    "losses": -0.7,
    "red": -0.4,
    "sell": -0.5,
    "selling": -0.55,
    "seller": -0.5,
    "sellers": -0.5,
    "plunge": -0.75,
    "plunging": -0.75,
    "breakdown": -0.7,
    "overvalued": -0.6,
    "collapse": -0.85,
    "collapsing": -0.85,
    "downside": -0.55,
    "panic": -0.8,
    "panicking": -0.8,
    "insolvent": -0.9,
    "insolvency": -0.9,
    "hack": -0.85,
    "hacked": -0.85,

    # General Negative
    "bad": -0.55,
    "terrible": -0.8,
    "worst": -0.85,
    "weak": -0.6,
    "weakness": -0.65,
    "poor": -0.6,
    "fail": -0.7,
    "failed": -0.75,
    "failure": -0.75,
    "danger": -0.7,
    "trouble": -0.6,
    "lose": -0.6,
    "losing": -0.65,
    "fear": -0.7,
    "negative": -0.6,
    "risk": -0.4,
    "risky": -0.5,

    # Sentiment Emojis
    "🚀": 0.8,
    "🔥": 0.5,
    "📈": 0.65,
    "💪": 0.5,
    "📉": -0.65,
    "🩸": -0.7,
    "💀": -0.75,
    "🚨": -0.4,
}

# Negation tokens triggering polarity inversion
NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "n't",
    "cannot",
    "hardly",
    "scarcely",
    "barely",
    "without",
    "lack",
    "lacking",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt",
    "won't",
    "wont",
    "isn't",
    "isnt",
    "aren't",
    "arent",
}

# Modifiers scaling subsequent token sentiment
INTENSIFIERS: Dict[str, float] = {
    "very": 1.3,
    "extremely": 1.4,
    "massively": 1.4,
    "huge": 1.3,
    "hugely": 1.3,
    "incredibly": 1.4,
    "highly": 1.3,
    "super": 1.3,
    "really": 1.25,
    "major": 1.25,
    "insanely": 1.4,
}

DIMINISHERS: Dict[str, float] = {
    "slightly": 0.7,
    "somewhat": 0.75,
    "barely": 0.6,
    "little": 0.7,
    "bit": 0.7,
    "mildly": 0.75,
}

# Subjectivity opinion markers for heuristic ratio computation
OPINION_MARKERS = {
    "think",
    "thinking",
    "thought",
    "believe",
    "believing",
    "feel",
    "feeling",
    "opinion",
    "imo",
    "imho",
    "prediction",
    "predict",
    "expect",
    "expecting",
    "guess",
    "speculate",
    "speculation",
    "probably",
    "maybe",
    "looks",
    "seems",
}


def _tokenize(text: str) -> List[str]:
    """
    Splits text into words, contractions, hashtags, and individual emojis.

    Args:
        text: Input string.

    Returns:
        List of lowercase token strings.
    """
    # Pattern captures words with apostrophes, hashtags, or individual non-whitespace characters (emojis/symbols)
    tokens = re.findall(r"#[A-Za-z0-9_]+|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s\w]", text.lower())
    return tokens


def compute_sentiment_score(text: Optional[str]) -> float:
    """
    Computes a deterministic sentiment polarity score bounded to [-1.0, 1.0].

    Features:
        - Safe neutral fallback (0.0) for None, empty, or whitespace-only text.
        - Idiomatic multi-word crypto phrase matching.
        - Negation handling within a 3-token window (flips polarity, dampened by 0.8x).
        - Valence scaling from intensifiers and diminishers.
        - Non-linear bounding via hyperbolic tangent: tanh(raw / 2.0).

    Args:
        text: Post text content.

    Returns:
        Sentiment polarity score in [-1.0, 1.0], rounded to 4 decimal places.
    """
    if text is None or not isinstance(text, str):
        return 0.0

    clean_text = text.strip()
    if not clean_text:
        return 0.0

    lower_text = clean_text.lower()
    raw_score = 0.0

    # 1. Evaluate and consume multi-word phrases (including preceding negations)
    text_for_tokens = lower_text
    phrase_negation_regex = r"(?:\b(not|no|never|hardly|barely|without|don't|dont|doesn't|doesnt|didn't|didnt|won't|wont|isn't|isnt|aren't|arent|cannot|do not)\b\s+)?"
    for phrase, phrase_weight in MULTI_WORD_PHRASES.items():
        if phrase in text_for_tokens:
            phrase_pattern = rf"{phrase_negation_regex}\b{re.escape(phrase)}\b"
            matches = list(re.finditer(phrase_pattern, text_for_tokens))
            for m in matches:
                neg = m.group(1)
                weight = -phrase_weight * 0.8 if neg else phrase_weight
                raw_score += weight
            # Mask out the entire matched span (negation + phrase) so tokens aren't duplicated
            text_for_tokens = re.sub(phrase_pattern, " ", text_for_tokens)

    tokens = _tokenize(text_for_tokens)
    if not tokens:
        bounded = math.tanh(raw_score / 2.0)
        return round(float(bounded), 4)

    # 2. Token-level evaluation with sliding negation & modifier window
    negation_counter = 0
    current_modifier = 1.0

    for i, token in enumerate(tokens):
        # Check for non-negating idiomatic exceptions such as "not only" or "not just"
        if token == "not" and i + 1 < len(tokens) and tokens[i + 1] in {"only", "just"}:
            continue

        # Check for negation triggers
        if token in NEGATION_TOKENS or token.endswith("n't"):
            negation_counter = 3
            current_modifier = 1.0
            continue

        # Check for intensifiers and diminishers
        if token in INTENSIFIERS:
            current_modifier = INTENSIFIERS[token]
            continue
        elif token in DIMINISHERS:
            current_modifier = DIMINISHERS[token]
            continue

        # Strip leading '#' if present for hashtag matching (e.g. #bullish -> bullish)
        stripped_token = token.lstrip("#")
        token_sentiment = SENTIMENT_LEXICON.get(stripped_token, SENTIMENT_LEXICON.get(token, None))

        if token_sentiment is not None:
            val = token_sentiment * current_modifier
            if negation_counter > 0:
                # Invert and dampen
                val = -val * 0.8
            raw_score += val
            current_modifier = 1.0

        if negation_counter > 0:
            negation_counter -= 1

    # Non-linear compression to strictly bound within [-1.0, 1.0]
    bounded = math.tanh(raw_score / 2.0)
    # Numerical safety clamp
    clamped = max(-1.0, min(1.0, float(bounded)))
    return round(clamped, 4)


def compute_subjectivity_score(text: Optional[str]) -> float:
    """
    Computes a heuristic subjectivity score bounded to [0.0, 1.0].

    Definition:
        Subjectivity is computed as the proportion of opinionated tokens
        (sentiment words, intensifiers, diminishers, and explicit opinion markers)
        relative to the total token count of the text.

    Important Notice:
        This is a lightweight lexical density heuristic designed for feature
        engineering and text profiling. It is NOT a clinically or scientifically
        validated psychological or cognitive measure.

    Args:
        text: Post text content.

    Returns:
        Subjectivity score in [0.0, 1.0], rounded to 4 decimal places.
    """
    if text is None or not isinstance(text, str):
        return 0.0

    clean_text = text.strip()
    if not clean_text:
        return 0.0

    tokens = _tokenize(clean_text)
    if not tokens:
        return 0.0

    # Filter out pure punctuation tokens to get meaningful word count
    word_tokens = [t for t in tokens if re.search(r"\w", t)]
    total_words = len(word_tokens)
    if total_words == 0:
        return 0.0

    subjective_count = 0
    for t in word_tokens:
        clean_t = t.lstrip("#")
        if (
            clean_t in SENTIMENT_LEXICON
            or clean_t in INTENSIFIERS
            or clean_t in DIMINISHERS
            or clean_t in OPINION_MARKERS
            or clean_t in NEGATION_TOKENS
        ):
            subjective_count += 1

    # Ratio bounded to [0.0, 1.0]
    ratio = min(1.0, subjective_count / total_words)
    return round(float(ratio), 4)


def classify_sentiment(
    score: float,
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
) -> str:
    """
    Categorizes a continuous sentiment score into discrete polarity labels.

    Args:
        score: Bounded sentiment score in [-1.0, 1.0].
        positive_threshold: Lower bound for 'positive' label.
        negative_threshold: Upper bound for 'negative' label.

    Returns:
        One of 'positive', 'negative', or 'neutral'.
    """
    if score > positive_threshold:
        return "positive"
    elif score < negative_threshold:
        return "negative"
    else:
        return "neutral"


def score_text(text: Optional[str]) -> Dict[str, Union[float, str]]:
    """
    Convenience function returning full sentiment and subjectivity metrics for a text.

    Args:
        text: Input string.

    Returns:
        Dictionary with keys:
            - sentiment_score (float in [-1.0, 1.0])
            - subjectivity_score (float in [0.0, 1.0])
            - sentiment_label (str: 'positive' | 'negative' | 'neutral')
    """
    sent = compute_sentiment_score(text)
    subj = compute_subjectivity_score(text)
    label = classify_sentiment(sent)
    return {
        "sentiment_score": sent,
        "subjectivity_score": subj,
        "sentiment_label": label,
    }


def score_posts(
    df: pd.DataFrame,
    text_column: str = "text",
) -> pd.DataFrame:
    """
    Applies deterministic sentiment scoring to a normalized posts DataFrame.

    Guarantees:
        - Keeps all original columns and rows completely intact.
        - Preserves identical row ordering and index.
        - Does not mutate engagement metrics or fabricate values.
        - Handles missing or empty text entries safely without crashing.
        - Appends 'sentiment_score', 'subjectivity_score', and 'sentiment_label'.

    Args:
        df: DataFrame following the normalized social post schema.
        text_column: Name of the text content column (default: 'text').

    Returns:
        New DataFrame with original fields plus sentiment columns.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        KeyError: If text_column is not present in df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if text_column not in df.columns:
        raise KeyError(f"Required text column '{text_column}' not found in DataFrame columns: {list(df.columns)}")

    # Operate on a copy to prevent in-place side effects
    result_df = df.copy()

    if result_df.empty:
        result_df["sentiment_score"] = pd.Series(dtype=float)
        result_df["subjectivity_score"] = pd.Series(dtype=float)
        result_df["sentiment_label"] = pd.Series(dtype=str)
        return result_df

    sentiment_scores: List[float] = []
    subjectivity_scores: List[float] = []
    sentiment_labels: List[str] = []

    for val in result_df[text_column]:
        if pd.isna(val):
            text_str = ""
        else:
            text_str = str(val)

        metrics = score_text(text_str)
        sentiment_scores.append(metrics["sentiment_score"])  # type: ignore
        subjectivity_scores.append(metrics["subjectivity_score"])  # type: ignore
        sentiment_labels.append(metrics["sentiment_label"])  # type: ignore

    result_df["sentiment_score"] = sentiment_scores
    result_df["subjectivity_score"] = subjectivity_scores
    result_df["sentiment_label"] = sentiment_labels

    return result_df


def load_synthetic_posts(
    fixture_path: Union[str, Path] = "tests/fixtures/synthetic_x_posts.json",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Utility loader for synthetic test fixtures.

    Args:
        fixture_path: Path to synthetic fixture JSON file.

    Returns:
        Tuple of (DataFrame of synthetic posts, metadata dictionary).

    Raises:
        FileNotFoundError: If fixture file does not exist.
        ValueError: If fixture does not contain synthetic confirmation metadata.
    """
    path = Path(fixture_path)
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic fixture not found at {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("_metadata", {})
    if not metadata.get("is_synthetic", False):
        raise ValueError(f"Fixture at {path} is not flagged with 'is_synthetic: true'")

    posts = data.get("posts", [])
    df = pd.DataFrame(posts)
    return df, metadata

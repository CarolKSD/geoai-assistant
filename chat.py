from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from ingest import (
    DEFAULT_STORAGE_DIR,
    IndexData,
    build_embedding_text,
    load_index,
    tokenize,
    vectorize_text,
)

DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODE = "hybrid-professor"
DEFAULT_STUDY_TASK = "ask"
DEFAULT_CANDIDATE_MULTIPLIER = 3
MAX_PER_FILE_CANDIDATES = 5
NEIGHBOR_WINDOW = 1
MAX_POSITIVE_CANDIDATES = 180
SOURCE_BOOST = 0.08
LECTURE_BOOST = 0.05
COURSE_BOOST = 0.08
PREFERRED_COURSE_BOOST = 0.14
INFERRED_COURSE_BOOST = 0.18
KEYWORD_BOOST = 0.18
PATH_KEYWORD_BOOST = 0.08
UNRELATED_COURSE_PENALTY = 0.07
CONCEPT_BOOST = 0.24
CONCEPT_TITLE_BOOST = 0.22
CONCEPT_SUPPORT_BOOST = 0.18
FRONT_MATTER_PENALTY = 0.22
SEMANTIC_SIMILARITY_WEIGHT = 0.55
RAW_SCORE_WEIGHT = 0.45
COURSE_CONFIDENCE_MEDIUM_THRESHOLD = 1.0
COURSE_CONFIDENCE_HIGH_THRESHOLD = 2.0
DOMINANT_COURSE_TOP_N = 3
FALLBACK_SCORE_RATIO_THRESHOLD = 0.5
QUESTION_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "all",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "between",
    "both",
    "briefly",
    "can",
    "concept",
    "define",
    "difference",
    "does",
    "each",
    "even",
    "explain",
    "for",
    "from",
    "give",
    "help",
    "how",
    "into",
    "its",
    "mean",
    "means",
    "more",
    "most",
    "plain",
    "show",
    "than",
    "that",
    "the",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}
CALCULATION_MARKERS = {
    "calculate",
    "calculated",
    "calculation",
    "compute",
    "computed",
    "formula",
    "formulas",
    "how",
}
CONTEXT_DEPENDENT_MARKERS = {
    "again",
    "also",
    "assumptions",
    "conditions",
    "continue",
    "difference",
    "formula",
    "how",
    "implications",
    "intuition",
    "its",
    "it",
    "more",
    "result",
    "same",
    "that",
    "theorem",
    "they",
    "this",
    "those",
    "what",
    "why",
}
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_USER_TURNS = 2
COURSE_HINT_KEYWORDS = {
    "adjustment theory": (
        "adjustment",
        "gauss-markov",
        "gauss markov",
        "blue",
        "best linear unbiased estimator",
        "propagation of observation errors",
        "propagation of errors",
        "error propagation",
        "propagation of variances",
        "variance covariance propagation",
        "jacobian",
        "functional model",
        "stochastic model",
        "least squares",
        "weighted least squares",
        "residual",
        "variance factor",
        "normal equation",
        "observation equation",
        "stochastic model",
    ),
    "geodatabases": (
        "geodatabases",
        "database",
        "sql",
        "postgis",
        "spatial index",
        "r-tree",
    ),
    "photogrametric cv": (
        "photogrammetry",
        "computer vision",
        "camera",
        "stereo",
        "matching",
        "sift",
        "bundle adjustment",
        "epipolar",
    ),
    "geoinformatics": (
        "geoinformatics",
        "raster",
        "vector",
        "gis",
        "dijkstra",
        "shortest path",
    ),
    "intr. space geodesy": (
        "space geodesy",
        "satellite",
        "orbit",
        "gnss",
    ),
}

CONCEPT_ALIAS_GROUPS = {
    "error propagation": (
        "propagation of observation errors",
        "propagation of observation error",
        "propagation of observations errors",
        "propagation of observational errors",
        "propagation of errors",
        "error propagation",
        "law of error propagation",
        "propagation of random deviations",
        "propagation of variances",
        "propagation of variances and covariances",
        "general law of propagation of variances",
        "variance covariance propagation",
        "variance-covariance propagation",
        "propagation law of variances and covariances",
    ),
}

CONCEPT_SUPPORT_TERMS = {
    "error propagation": (
        "functional model",
        "stochastic model",
        "jacobian",
        "design matrix",
        "partial derivative",
        "partial derivatives",
        "variance covariance matrix",
        "variance-covariance matrix",
        "covariance matrix",
        "variance covariance propagation",
        "variance-covariance propagation",
        "propagation of variances and covariances",
        "sigma_ff",
        "sigma_ll",
        "standard deviation",
        "covariance",
        "variance",
    ),
}

CONCEPT_APPLICATION_TERMS = {
    "error propagation": (
        "traverse",
        "misclosure",
        "polygon traverse",
        "link traverse",
        "leveling",
        "benchmark",
        "cartesian coordinates",
        "coordinate transformation",
        "edm",
        "azimuth",
        "departure",
        "latitude",
    ),
}

def build_system_prompt(
    mode: str,
    study_task: str = DEFAULT_STUDY_TASK,
    beginner_mode: bool = False,
) -> str:
    shared_prompt = (
        "You are a local study assistant for a Master's in Geodesy and Geoinformation Science. "
        "Write clear, natural, academically strong explanations rather than fragmented slide text or stiff prose. "
        "For theoretical concepts, aim for this structure when the material supports it: definition, core components, assumptions or conditions, result or implication, and a short intuition if useful. "
        "When formulas appear, explain them in plain language and define symbols when possible. "
        "Never hallucinate formulas, citations, or claims. "
        "Never contradict the retrieved local study materials. "
        "Stay strictly aligned with the exact concept asked in the question. "
        "If the user asks a direct concept question, answer it directly. Do not reply with a meta-response such as asking what the user wants to do with the material. "
        "Do not change the topic, do not reinterpret the question, and do not replace the target concept with a related one. "
        "If retrieved material is relevant but incomplete, do NOT switch to another concept. Reconstruct the same concept from the available material and extend it carefully. "
    )

    if study_task == "summarize-lecture":
        task_prompt = (
            "When the user asks for a lecture summary, write a coherent study summary rather than a list of slide fragments. "
            "Organize it with a short overview, the main concepts, the important formulas or methods if relevant, and the main takeaways for revision. "
            "Merge overlapping slide content into one clean explanation. "
        )
    elif study_task == "generate-exam-questions":
        task_prompt = (
            "When the user asks for exam questions, generate 5 to 8 exam-style questions grounded in the retrieved material. "
            "Mix definition, interpretation, derivation, and application questions when the material supports them. "
            "After each question, give a short expected-answer guide. "
        )
    else:
        task_prompt = (
            "When the user asks for an explanation, structure the answer clearly and make it easy to study from. "
        )

    audience_prompt = ""
    if beginner_mode:
        audience_prompt = (
            "The user enabled beginner mode. Use simpler language, define jargon when it first appears, prefer shorter sentences, and add brief intuition where it helps. "
            "Do not become vague or childish; keep the explanation technically correct. "
        )

    if mode == "strict":
        return (
            shared_prompt
            + task_prompt
            + audience_prompt
            + "The retrieved local study materials are the only allowed source. "
            + "Do not use outside or general knowledge. "
            + "If a term is not explicitly named but can be reconstructed from related concepts in the retrieved materials, you may infer that link, but only from the local material itself. "
            + "If the question is about a core Adjustment Theory concept such as the Gauss-Markov model, least squares, or the stochastic model, attempt reconstruction from the retrieved functional-model and stochastic-model material rather than drifting to another topic. "
            + "Only say the concept is not found when the retrieved material is genuinely unrelated to that concept."
        )

    return (
        shared_prompt
        + task_prompt
        + audience_prompt
        + "The retrieved local study materials are the primary source and must anchor the answer. "
        + "If relevant course material is retrieved, you MUST base your answer on it and stay within that topic. "
        + "You are NOT allowed to switch to a different concept or reinterpret the question. "
        + "If the material is incomplete, extend it using careful general knowledge, but keep the same concept and terminology. "
        + "If only weak signals exist, still answer the same concept using standard theory rather than replacing it with another topic. "
        + "For core Adjustment Theory concepts such as the Gauss-Markov model, least squares, and the stochastic model, attempt reconstruction using the functional model, stochastic model, and standard terminology when the retrieved material supports those links. "
        + "Do not change domain: if the retrieved context is clearly from one course, stay in that course domain unless another domain is directly relevant to the same concept. "
        + "Any general explanatory addition must be signaled briefly with phrases such as "
        + "\"Based on the course materials...\", \"More generally...\", or \"In standard terminology...\". "
        + "Do not present secondary general knowledge as if it came from the local files, and do not attach local citations to it. "
        + "If an exact term is not explicitly defined but can be reconstructed from the course material and standard terminology, do that carefully and confidently. "
        + "Be especially confident when the course context links the functional model and stochastic model; in this material, that often supports a complete explanation of the Gauss-Markov model, including its theoretical meaning and BLUE-type result. "
        + "Only say the concept is not found when the retrieved material is genuinely unrelated to that concept and careful standard knowledge still cannot connect it."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against the local study-material index.")
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE_DIR,
        help="Directory that contains the index created by src/ingest.py.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name for answer generation.",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Base URL for the local Ollama API.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of merged context passages to send to the model.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.015,
        help="Minimum cosine similarity score before retrieval falls back to weaker positive matches.",
    )
    parser.add_argument(
        "--question",
        help="Optional one-shot question. If omitted, an interactive CLI session starts.",
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "hybrid-professor"],
        default=DEFAULT_MODE,
        help="Answer mode: 'strict' uses only local materials, 'hybrid-professor' uses local materials first and careful general knowledge second.",
    )
    parser.add_argument(
        "--task",
        choices=["ask", "summarize-lecture", "generate-exam-questions"],
        default=DEFAULT_STUDY_TASK,
        help="Study task: ask a question, summarize a lecture/topic, or generate exam questions.",
    )
    parser.add_argument(
        "--beginner",
        action="store_true",
        help="Use simpler explanations with less jargon while keeping the content technically correct.",
    )
    parser.add_argument(
        "--course",
        action="append",
        default=[],
        help="Optional course-name filter. Can be provided multiple times and matches case-insensitive substrings.",
    )
    parser.add_argument(
        "--folder",
        action="append",
        default=[],
        help="Optional folder/path filter. Can be provided multiple times and matches case-insensitive substrings.",
    )
    parser.add_argument(
        "--show-retrieved",
        action="store_true",
        help="Print retrieved passages before the answer for debugging.",
    )
    return parser.parse_args()


def get_page_range(chunk: dict[str, Any]) -> tuple[int | None, int | None]:
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    page = chunk.get("page")

    if page_start is None and page is not None:
        page_start = int(page)
    if page_end is None and page_start is not None:
        page_end = int(page_start)

    return (
        int(page_start) if page_start is not None else None,
        int(page_end) if page_end is not None else None,
    )


def format_page_range(page_start: int | None, page_end: int | None) -> str | None:
    if page_start is None:
        return None
    if page_end is None or page_start == page_end:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"


def lecture_key_for_relpath(source_relpath: str) -> str:
    path = Path(source_relpath)
    stem = re.sub(r"(?i)(?:[_\-\s]?part[_\-\s]?\d+)$", "", path.stem).strip("_- ")
    return str(path.with_name(stem or path.stem))


def course_name_for_relpath(source_relpath: str) -> str:
    parts = Path(source_relpath).parts
    if len(parts) >= 2 and "semester" in parts[0].lower():
        return parts[1]
    if parts:
        return parts[0]
    return source_relpath


def normalize_text_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    normalized = re.sub(r"^[^\w]+|[^\w]+$", "", normalized)
    return normalized


def contains_substring_match(text: str, patterns: list[str]) -> bool:
    normalized_text = normalize_text_key(text)
    return any(pattern in normalized_text for pattern in patterns)


def normalize_filters(values: list[str]) -> list[str]:
    normalized = [normalize_text_key(value) for value in values if normalize_text_key(value)]
    return list(dict.fromkeys(normalized))


def normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.lower())).strip()


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_question_terms(question: str) -> list[str]:
    terms: list[str] = []
    for token in tokenize(question):
        normalized = normalize_text_key(token)
        if len(normalized) < 3 or normalized in QUESTION_STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def infer_concept_profile(question: str) -> dict[str, Any]:
    lower_question = question.lower()
    match_question = normalize_match_text(question)
    concept_scores: dict[str, float] = {}
    concept_aliases: dict[str, list[str]] = {}

    for concept_key, aliases in CONCEPT_ALIAS_GROUPS.items():
        score = 0.0
        matched_aliases: list[str] = []
        normalized_concept_key = normalize_match_text(concept_key)
        if concept_key in lower_question or normalized_concept_key in match_question:
            score += 1.6
            matched_aliases.append(concept_key)

        for alias in aliases:
            normalized_alias = normalize_match_text(alias)
            if alias in lower_question or normalized_alias in match_question:
                score += 1.2 if " " in alias else 1.0
                matched_aliases.append(alias)

        if score > 0:
            concept_scores[concept_key] = score
            concept_aliases[concept_key] = list(dict.fromkeys(matched_aliases or [concept_key]))

    target_concept_key = None
    target_concept_score = 0.0
    target_concept_aliases: list[str] = []
    if concept_scores:
        target_concept_key, target_concept_score = max(
            concept_scores.items(),
            key=lambda item: item[1],
        )
        target_concept_aliases = list(
            dict.fromkeys(
                [target_concept_key]
                + list(CONCEPT_ALIAS_GROUPS.get(target_concept_key, ()))
                + concept_aliases.get(target_concept_key, [])
            )
        )

    return {
        "concept_scores": concept_scores,
        "target_concept_key": target_concept_key,
        "target_concept_score": target_concept_score,
        "target_concept_aliases": target_concept_aliases,
    }


def build_retrieval_query(question: str, profile: dict[str, Any]) -> str:
    aliases = list(profile.get("target_concept_aliases") or [])
    if not aliases:
        return question

    canonical = profile.get("target_concept_key")
    expansion_parts = []
    if canonical:
        expansion_parts.append(f"canonical concept: {canonical}")
    if aliases:
        expansion_parts.append("related terms: " + ", ".join(aliases[:8]))
    if profile.get("asks_for_calculation") and canonical in CONCEPT_SUPPORT_TERMS:
        support_terms = CONCEPT_SUPPORT_TERMS[canonical]
        expansion_parts.append("calculation terms: " + ", ".join(support_terms[:10]))

    if not expansion_parts:
        return question
    return question + "\n" + "\n".join(expansion_parts)


def question_requests_calculation(question: str) -> bool:
    normalized = normalize_match_text(question)
    if not normalized:
        return False

    if "how is it calculated" in normalized or "how it is calculated" in normalized:
        return True
    if "how is it computed" in normalized or "how it is computed" in normalized:
        return True
    if "how do you calculate" in normalized or "how to calculate" in normalized:
        return True

    return any(token in CALCULATION_MARKERS for token in normalized.split())


def candidate_front_matter_penalty(candidate: dict[str, Any]) -> float:
    title = normalize_match_text(str(candidate["chunk"].get("title", "")))
    text = normalize_match_text(str(candidate["chunk"].get("text", ""))[:1200])
    combined = f"{title} {text}".strip()
    if not combined:
        return 0.0

    penalty = 0.0
    if "copyright notice" in combined:
        penalty += 0.16
    if title == "content" or title == "contents":
        penalty += 0.12
    if text.startswith("contents ") or text == "contents":
        penalty += 0.18
    if "this provided digital course material is protected" in combined:
        penalty += 0.10
    if "table of contents" in combined:
        penalty += 0.10

    return min(FRONT_MATTER_PENALTY, penalty)


def build_query_profile(question: str, course_filters: list[str], folder_filters: list[str]) -> dict[str, Any]:
    lower_question = question.lower()
    match_question = normalize_match_text(question)
    key_terms = extract_question_terms(question)
    concept_profile = infer_concept_profile(question)
    asks_for_calculation = question_requests_calculation(question)
    course_hint_scores: dict[str, float] = {}

    for course_key, keywords in COURSE_HINT_KEYWORDS.items():
        score = 0.0
        normalized_course_key = normalize_match_text(course_key)
        if course_key in lower_question or normalized_course_key in match_question:
            score += 2.0
        for keyword in keywords:
            normalized_keyword = normalize_match_text(keyword)
            if keyword in lower_question or normalized_keyword in match_question:
                score += 1.2 if " " in keyword else 1.0
        if score > 0:
            course_hint_scores[course_key] = score

    inferred_course_key = None
    inferred_course_score = 0.0
    if course_hint_scores:
        inferred_course_key, inferred_course_score = max(
            course_hint_scores.items(),
            key=lambda item: item[1],
        )
    inferred_course_confidence = classify_inferred_course_confidence(inferred_course_score)

    return {
        "question_lower": lower_question,
        "key_terms": key_terms,
        "course_filters": normalize_filters(course_filters),
        "folder_filters": normalize_filters(folder_filters),
        "course_hint_scores": course_hint_scores,
        "inferred_course_key": inferred_course_key,
        "inferred_course_score": inferred_course_score,
        "inferred_course_confidence": inferred_course_confidence,
        "target_concept_key": concept_profile["target_concept_key"],
        "target_concept_score": float(concept_profile["target_concept_score"]),
        "target_concept_aliases": concept_profile["target_concept_aliases"],
        "asks_for_calculation": asks_for_calculation,
    }


def classify_inferred_course_confidence(score: float) -> str:
    if score >= COURSE_CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if score >= COURSE_CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def question_needs_conversation_context(question: str) -> bool:
    normalized = normalize_match_text(question)
    if not normalized:
        return False

    tokens = normalized.split()
    if len(tokens) <= 6 and any(token in CONTEXT_DEPENDENT_MARKERS for token in tokens):
        return True

    leading_phrases = (
        "and ",
        "what about",
        "how about",
        "and what",
        "and how",
        "can you expand",
        "can you explain more",
    )
    if any(normalized.startswith(phrase) for phrase in leading_phrases):
        return True

    return any(token in CONTEXT_DEPENDENT_MARKERS for token in tokens[:3])


def recent_user_turns(conversation_history: list[dict[str, Any]], max_turns: int = MAX_HISTORY_USER_TURNS) -> list[str]:
    user_turns: list[str] = []
    for item in reversed(conversation_history):
        if item.get("role") != "user":
            continue
        content = compact_whitespace(str(item.get("content", "")))
        if not content:
            continue
        user_turns.append(content)
        if len(user_turns) >= max_turns:
            break
    user_turns.reverse()
    return user_turns


def build_effective_question(
    question: str,
    conversation_history: list[dict[str, Any]] | None,
) -> str:
    if not conversation_history or not question_needs_conversation_context(question):
        return question

    previous_turns = recent_user_turns(conversation_history, max_turns=MAX_HISTORY_USER_TURNS)
    if not previous_turns:
        return question

    context_lines = [f"- {turn}" for turn in previous_turns]
    return (
        "Recent conversation context:\n"
        + "\n".join(context_lines)
        + f"\nCurrent question:\n{question}"
    )


def format_conversation_history(
    conversation_history: list[dict[str, Any]] | None,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> str:
    if not conversation_history:
        return ""

    recent_messages = conversation_history[-max_messages:]
    lines: list[str] = []
    for item in recent_messages:
        role = "User" if item.get("role") == "user" else "Assistant"
        content = compact_whitespace(str(item.get("content", "")))
        if not content:
            continue
        lines.append(f"{role}: {content}")

    if not lines:
        return ""

    return (
        "Recent conversation history for continuity only:\n"
        + "\n".join(lines)
        + "\nUse this only to resolve references from the latest user request.\n\n"
    )


def split_text_units(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if "\n" in text:
        return [line.strip() for line in text.splitlines() if line.strip()]
    return [unit.strip() for unit in re.split(r"(?<=[.!?])\s+", text) if unit.strip()]


def deduplicate_text(text: str, seen_units: set[str]) -> str:
    units = split_text_units(text)
    if not units:
        return ""

    deduplicated_units: list[str] = []
    for unit in units:
        normalized = normalize_text_key(unit)
        if not normalized or normalized in seen_units:
            continue
        seen_units.add(normalized)
        deduplicated_units.append(unit)

    if not deduplicated_units:
        return ""

    separator = "\n" if "\n" in text else " "
    return separator.join(deduplicated_units)


def chunk_sort_key(chunk: dict[str, Any]) -> tuple[int, int, int, int]:
    page_start, _ = get_page_range(chunk)
    if page_start is not None:
        return (
            0,
            page_start,
            int(chunk.get("chunk_number", 1)),
            int(chunk.get("chunk_id", 0)),
        )

    return (
        1,
        int(chunk.get("segment_index", 1)),
        int(chunk.get("chunk_number", 1)),
        int(chunk.get("chunk_id", 0)),
    )


def chunks_are_close(left_chunk: dict[str, Any], right_chunk: dict[str, Any]) -> bool:
    if left_chunk["source_relpath"] != right_chunk["source_relpath"]:
        return False

    if chunk_sort_key(left_chunk) > chunk_sort_key(right_chunk):
        left_chunk, right_chunk = right_chunk, left_chunk

    left_page_start, left_page_end = get_page_range(left_chunk)
    right_page_start, _ = get_page_range(right_chunk)
    if left_page_start is not None and right_page_start is not None:
        if left_page_start == right_page_start:
            return abs(
                int(left_chunk.get("chunk_number", 1)) - int(right_chunk.get("chunk_number", 1))
            ) <= 1
        return 0 < right_page_start - int(left_page_end or left_page_start) <= 1

    left_segment_index = int(left_chunk.get("segment_index", 1))
    right_segment_index = int(right_chunk.get("segment_index", 1))
    if left_segment_index == right_segment_index:
        return abs(
            int(left_chunk.get("chunk_number", 1)) - int(right_chunk.get("chunk_number", 1))
        ) <= 1

    return (
        0 < right_segment_index - left_segment_index <= 1
        and int(right_chunk.get("chunk_number", 1)) == 1
    )


def build_source_orders(index: IndexData) -> tuple[dict[str, list[int]], dict[int, int]]:
    ordered_by_source: dict[str, list[int]] = {}
    positions: dict[int, int] = {}

    for raw_index, chunk in enumerate(index.chunks):
        ordered_by_source.setdefault(chunk["source_relpath"], []).append(raw_index)

    for source_relpath, raw_indices in ordered_by_source.items():
        raw_indices.sort(key=lambda item: chunk_sort_key(index.chunks[item]))
        ordered_by_source[source_relpath] = raw_indices
        for position, raw_index in enumerate(raw_indices):
            positions[raw_index] = position

    return ordered_by_source, positions


def candidate_matches_filters(
    candidate: dict[str, Any],
    course_filters: list[str],
    folder_filters: list[str],
) -> bool:
    if course_filters:
        course_haystacks = [
            candidate["course_name"],
            candidate["source_relpath"],
            candidate.get("source_path", ""),
        ]
        if not any(
            contains_substring_match(haystack, course_filters) for haystack in course_haystacks if haystack
        ):
            return False

    if folder_filters:
        folder_haystacks = [candidate["source_relpath"], candidate.get("source_path", "")]
        if not any(
            contains_substring_match(haystack, folder_filters) for haystack in folder_haystacks if haystack
        ):
            return False

    return True


def candidate_course_keys(candidate: dict[str, Any]) -> list[str]:
    haystacks = [
        candidate["course_name"],
        candidate["source_relpath"],
        candidate.get("source_path", ""),
    ]
    matches: list[str] = []

    for course_key in COURSE_HINT_KEYWORDS:
        if any(contains_substring_match(haystack, [normalize_text_key(course_key)]) for haystack in haystacks if haystack):
            matches.append(course_key)

    return matches


def entity_matches_course_key(entity: dict[str, Any], course_key: str | None) -> bool:
    if not course_key:
        return False
    return course_key in candidate_course_keys(entity)


def summarize_group_strength(
    candidates: list[dict[str, Any]],
    key_name: str,
    top_n: int,
) -> dict[str, float]:
    grouped_scores: dict[str, list[float]] = {}
    for candidate in candidates:
        grouped_scores.setdefault(candidate[key_name], []).append(float(candidate["raw_score"]))

    return {
        key: sum(sorted(scores, reverse=True)[:top_n])
        for key, scores in grouped_scores.items()
    }


def build_candidate_text(candidate: dict[str, Any]) -> str:
    parts = [
        candidate["course_name"],
        candidate["source_relpath"],
        candidate.get("source_path", ""),
        build_embedding_text(candidate["chunk"]),
    ]
    return "\n".join(part for part in parts if part).strip()


def compute_keyword_overlap(key_terms: list[str], text: str) -> tuple[float, int]:
    if not key_terms:
        return 0.0, 0

    lower_text = text.lower()
    token_set = set(tokenize(text))
    hits = 0
    for term in key_terms:
        if term in token_set or term in lower_text:
            hits += 1

    return hits / len(key_terms), hits


def compute_concept_match_score(candidate: dict[str, Any], concept_aliases: list[str]) -> tuple[float, float, int]:
    if not concept_aliases:
        return 0.0, 0.0, 0

    title_text = normalize_match_text(str(candidate["chunk"].get("title", "")))
    body_text = normalize_match_text(build_candidate_text(candidate))
    score = 0.0
    title_score = 0.0
    hits = 0

    for alias in concept_aliases:
        normalized_alias = normalize_match_text(alias)
        if not normalized_alias:
            continue

        weight = 1.0 if " " in alias else 0.7
        if normalized_alias in body_text:
            score += weight
            hits += 1
        if title_text and normalized_alias in title_text:
            title_score = max(title_score, weight)

    return min(1.0, score / 2.4), min(1.0, title_score), hits


def compute_support_term_score(
    candidate: dict[str, Any],
    support_terms: list[str],
) -> tuple[float, int]:
    if not support_terms:
        return 0.0, 0

    text = normalize_match_text(build_candidate_text(candidate))
    if not text:
        return 0.0, 0

    hits = 0
    for term in support_terms:
        normalized_term = normalize_match_text(term)
        if normalized_term and normalized_term in text:
            hits += 1

    if hits <= 0:
        return 0.0, 0
    return min(1.0, hits / 4.0), hits


def question_course_bonus(course_name: str, source_relpath: str, profile: dict[str, Any]) -> float:
    matched_course_keys = []
    candidate = {
        "course_name": course_name,
        "source_relpath": source_relpath,
        "source_path": source_relpath,
    }
    matched_course_keys = candidate_course_keys(candidate)
    bonus = 0.0

    for course_key in matched_course_keys:
        hint_score = float(profile["course_hint_scores"].get(course_key, 0.0))
        if hint_score > 0:
            bonus += 0.04 * min(hint_score, 3.0)

    inferred_course_key = profile["inferred_course_key"]
    inferred_course_score = float(profile["inferred_course_score"])
    inference_strength = min(1.0, inferred_course_score / 3.0) if inferred_course_score > 0 else 0.0

    if inferred_course_key:
        if inferred_course_key in matched_course_keys:
            bonus += INFERRED_COURSE_BOOST * inference_strength
        elif matched_course_keys:
            bonus -= UNRELATED_COURSE_PENALTY * inference_strength

    return bonus


def pick_preferred_course(
    course_strength: dict[str, float],
    profile: dict[str, Any],
) -> str | None:
    if not course_strength:
        return None

    course_items = sorted(course_strength.items(), key=lambda item: item[1], reverse=True)

    if profile["course_filters"]:
        for course_name, _ in course_items:
            if contains_substring_match(course_name, profile["course_filters"]):
                return course_name

    inferred_course_key = profile["inferred_course_key"]
    if inferred_course_key:
        for course_name, _ in course_items:
            candidate = {
                "course_name": course_name,
                "source_relpath": course_name,
                "source_path": course_name,
            }
            if inferred_course_key in candidate_course_keys(candidate):
                return course_name

    for course_name, _ in course_items:
        if contains_substring_match(profile["question_lower"], [normalize_text_key(course_name)]):
            return course_name

    return course_items[0][0]


def rerank_candidates(
    question_vector: np.ndarray,
    dimensions: int,
    idf: np.ndarray,
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    source_strength = summarize_group_strength(candidates[:80], "source_relpath", top_n=3)
    lecture_strength = summarize_group_strength(candidates[:120], "lecture_key", top_n=4)
    course_strength = summarize_group_strength(candidates[:120], "course_name", top_n=4)
    preferred_course = pick_preferred_course(course_strength, profile)
    concept_aliases = list(profile.get("target_concept_aliases") or [])
    support_terms = []
    if profile.get("asks_for_calculation") and profile.get("target_concept_key") in CONCEPT_SUPPORT_TERMS:
        support_terms = list(CONCEPT_SUPPORT_TERMS[profile["target_concept_key"]])

    for candidate in candidates:
        candidate_text = build_candidate_text(candidate)
        rerank_vector = vectorize_text(candidate_text, dimensions=dimensions, idf=idf)
        semantic_similarity = float(question_vector @ rerank_vector) if np.any(rerank_vector) else 0.0
        keyword_overlap, keyword_hits = compute_keyword_overlap(profile["key_terms"], candidate_text)
        concept_match, concept_title_match, concept_hits = compute_concept_match_score(
            candidate,
            concept_aliases,
        )
        support_term_score, support_term_hits = compute_support_term_score(candidate, support_terms)
        path_overlap, path_hits = compute_keyword_overlap(
            profile["key_terms"],
            f"{candidate['course_name']} {candidate['source_relpath']}",
        )

        course_bonus = question_course_bonus(
            candidate["course_name"],
            candidate["source_relpath"],
            profile,
        )
        if preferred_course and candidate["course_name"] == preferred_course:
            course_bonus += PREFERRED_COURSE_BOOST

        candidate["semantic_similarity"] = semantic_similarity
        candidate["keyword_overlap"] = keyword_overlap
        candidate["keyword_hits"] = keyword_hits
        candidate["concept_match"] = concept_match
        candidate["concept_title_match"] = concept_title_match
        candidate["concept_hits"] = concept_hits
        candidate["support_term_score"] = support_term_score
        candidate["support_term_hits"] = support_term_hits
        candidate["preferred_course"] = preferred_course
        penalty = candidate_front_matter_penalty(candidate)
        candidate["score"] = (
            RAW_SCORE_WEIGHT * float(candidate["raw_score"])
            + SEMANTIC_SIMILARITY_WEIGHT * semantic_similarity
            + KEYWORD_BOOST * keyword_overlap
            + CONCEPT_BOOST * concept_match
            + CONCEPT_TITLE_BOOST * concept_title_match
            + CONCEPT_SUPPORT_BOOST * support_term_score
            + PATH_KEYWORD_BOOST * path_overlap
            + SOURCE_BOOST * min(1.0, source_strength.get(candidate["source_relpath"], 0.0))
            + LECTURE_BOOST * min(1.0, lecture_strength.get(candidate["lecture_key"], 0.0))
            + COURSE_BOOST * min(1.0, course_strength.get(candidate["course_name"], 0.0))
            + course_bonus
            - penalty
        )
        candidate["path_keyword_hits"] = path_hits
        candidate["front_matter_penalty"] = penalty

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["support_term_hits"],
            item["concept_hits"],
            item["semantic_similarity"],
            item["keyword_hits"],
            item["raw_score"],
        ),
        reverse=True,
    )
    return candidates


def retrieve_candidate_chunks(
    question: str,
    index: IndexData,
    top_k: int,
    min_score: float,
    course_filters: list[str],
    folder_filters: list[str],
) -> list[dict[str, Any]]:
    dimensions = int(index.config["embedding_dimensions"])
    profile = build_query_profile(question, course_filters=course_filters, folder_filters=folder_filters)
    retrieval_query = build_retrieval_query(question, profile)
    query_vector = vectorize_text(retrieval_query, dimensions=dimensions, idf=index.idf)
    if not np.any(query_vector):
        return []

    scores = index.embeddings @ query_vector
    ranked_indices = np.argsort(scores)[::-1]

    candidate_limit = max(top_k * DEFAULT_CANDIDATE_MULTIPLIER, top_k)
    positive_candidates: list[dict[str, Any]] = []

    for raw_index in ranked_indices:
        score = float(scores[int(raw_index)])
        if score <= 0:
            break

        chunk = index.chunks[int(raw_index)]
        candidate = {
            "raw_index": int(raw_index),
            "raw_score": score,
            "score": score,
            "chunk": chunk,
            "source_relpath": chunk["source_relpath"],
            "source_path": chunk.get("source_path", chunk["source_relpath"]),
            "course_name": course_name_for_relpath(chunk["source_relpath"]),
            "lecture_key": lecture_key_for_relpath(chunk["source_relpath"]),
        }
        candidate["matched_course_keys"] = candidate_course_keys(candidate)
        if candidate_matches_filters(
            candidate,
            course_filters=profile["course_filters"],
            folder_filters=profile["folder_filters"],
        ):
            positive_candidates.append(candidate)

        if len(positive_candidates) >= MAX_POSITIVE_CANDIDATES:
            break

    if not positive_candidates:
        return []
    reranked_candidates = rerank_candidates(
        question_vector=query_vector,
        dimensions=dimensions,
        idf=index.idf,
        candidates=positive_candidates,
        profile=profile,
    )

    strong_results: list[dict[str, Any]] = []
    fallback_results: list[dict[str, Any]] = []
    per_file_counts: dict[str, int] = {}
    fallback_limit = max(top_k, 3)

    for candidate in reranked_candidates:
        source_relpath = candidate["source_relpath"]
        if per_file_counts.get(source_relpath, 0) >= MAX_PER_FILE_CANDIDATES:
            continue

        per_file_counts[source_relpath] = per_file_counts.get(source_relpath, 0) + 1

        if float(candidate["raw_score"]) >= min_score:
            strong_results.append(candidate)
        elif len(fallback_results) < fallback_limit:
            fallback_results.append(candidate)

        if len(strong_results) >= candidate_limit:
            break

    if not strong_results:
        return fallback_results

    if len(strong_results) < top_k:
        strong_results.extend(fallback_results[: max(0, top_k - len(strong_results))])

    return strong_results


def build_passage(
    index: IndexData,
    raw_indices: list[int],
    support_scores: dict[int, float],
) -> dict[str, Any]:
    chunks = [index.chunks[raw_index] for raw_index in raw_indices]
    titles: list[str] = []
    parts: list[str] = []
    last_header: str | None = None
    page_starts: list[int] = []
    page_ends: list[int] = []
    seen_units: set[str] = set()

    for chunk in chunks:
        page_start, page_end = get_page_range(chunk)
        if page_start is not None:
            page_starts.append(page_start)
            page_ends.append(int(page_end or page_start))

        title = chunk.get("title")
        if title and title not in titles:
            titles.append(title)

        header_parts = []
        page_label = format_page_range(page_start, page_end)
        if page_label:
            header_parts.append(page_label)
        if title:
            header_parts.append(title)
        header = " | ".join(header_parts)
        body = deduplicate_text(chunk.get("text", "").strip(), seen_units)

        if body:
            if header and header != last_header:
                parts.append(f"{header}\n{body}")
            else:
                parts.append(body)
        elif header and header != last_header:
            parts.append(header)

        if header:
            last_header = header

    return {
        "score": max(support_scores[raw_index] for raw_index in raw_indices)
        + (0.01 * max(0, len(raw_indices) - 1)),
        "source_relpath": chunks[0]["source_relpath"],
        "source_path": chunks[0].get("source_path", chunks[0]["source_relpath"]),
        "course_name": course_name_for_relpath(chunks[0]["source_relpath"]),
        "lecture_key": lecture_key_for_relpath(chunks[0]["source_relpath"]),
        "page_start": min(page_starts) if page_starts else None,
        "page_end": max(page_ends) if page_ends else None,
        "titles": titles,
        "text": "\n\n".join(parts).strip(),
    }


def passage_relevance_adjustment(passage: dict[str, Any], profile: dict[str, Any]) -> float:
    title_text = normalize_match_text(" ".join(passage.get("titles") or []))
    body_text = normalize_match_text(
        "\n".join(
            [
                passage.get("course_name", ""),
                passage.get("source_relpath", ""),
                passage.get("text", ""),
            ]
        )
    )

    concept_bonus = 0.0
    concept_aliases = list(profile.get("target_concept_aliases") or [])
    if concept_aliases:
        hits = 0
        title_hits = 0
        for alias in concept_aliases:
            normalized_alias = normalize_match_text(alias)
            if not normalized_alias:
                continue
            if normalized_alias in body_text:
                hits += 1
            if title_text and normalized_alias in title_text:
                title_hits += 1
        concept_bonus += 0.05 * min(hits, 4)
        concept_bonus += 0.04 * min(title_hits, 2)

    if profile.get("asks_for_calculation") and profile.get("target_concept_key") in CONCEPT_SUPPORT_TERMS:
        support_hits = 0
        for term in CONCEPT_SUPPORT_TERMS[profile["target_concept_key"]]:
            normalized_term = normalize_match_text(term)
            if normalized_term and normalized_term in body_text:
                support_hits += 1
        concept_bonus += 0.035 * min(support_hits, 4)

    penalty = 0.0
    if body_text.startswith("copyright notice") or "this provided digital course material is protected" in body_text:
        penalty += 0.14
    if body_text.startswith("contents ") or "\ncontents " in body_text:
        penalty += 0.12

    target_concept_key = profile.get("target_concept_key")
    if target_concept_key in CONCEPT_APPLICATION_TERMS:
        question_text = normalize_match_text(str(profile.get("question_lower", "")))
        application_hits = 0
        for term in CONCEPT_APPLICATION_TERMS[target_concept_key]:
            normalized_term = normalize_match_text(term)
            if not normalized_term or normalized_term in question_text:
                continue
            if normalized_term in body_text:
                application_hits += 1
        penalty += 0.05 * min(application_hits, 3)

    return concept_bonus - penalty


def passage_concept_priority(passage: dict[str, Any], profile: dict[str, Any]) -> float:
    target_concept_key = profile.get("target_concept_key")
    if not target_concept_key:
        return 0.0

    title_text = normalize_match_text(" ".join(passage.get("titles") or []))
    body_text = normalize_match_text(
        "\n".join(
            [
                passage.get("course_name", ""),
                passage.get("source_relpath", ""),
                passage.get("text", ""),
            ]
        )
    )

    alias_hits = 0
    for alias in profile.get("target_concept_aliases") or []:
        normalized_alias = normalize_match_text(alias)
        if not normalized_alias:
            continue
        if normalized_alias in body_text or normalized_alias in title_text:
            alias_hits += 1

    support_hits = 0
    for term in CONCEPT_SUPPORT_TERMS.get(target_concept_key, ()):
        normalized_term = normalize_match_text(term)
        if normalized_term and normalized_term in body_text:
            support_hits += 1

    application_hits = 0
    question_text = normalize_match_text(str(profile.get("question_lower", "")))
    for term in CONCEPT_APPLICATION_TERMS.get(target_concept_key, ()):
        normalized_term = normalize_match_text(term)
        if not normalized_term or normalized_term in question_text:
            continue
        if normalized_term in body_text:
            application_hits += 1

    return float(alias_hits) + (0.45 * support_hits) - (0.75 * application_hits)


def merge_context_passages(
    index: IndexData,
    support_scores: dict[int, float],
) -> list[dict[str, Any]]:
    if not support_scores:
        return []

    sorted_indices = sorted(
        support_scores,
        key=lambda raw_index: (
            index.chunks[raw_index]["source_relpath"],
            chunk_sort_key(index.chunks[raw_index]),
        ),
    )

    grouped_indices: list[list[int]] = []
    current_group = [sorted_indices[0]]

    for raw_index in sorted_indices[1:]:
        previous_chunk = index.chunks[current_group[-1]]
        current_chunk = index.chunks[raw_index]
        if chunks_are_close(previous_chunk, current_chunk):
            current_group.append(raw_index)
            continue

        grouped_indices.append(current_group)
        current_group = [raw_index]

    grouped_indices.append(current_group)
    return [build_passage(index, group, support_scores) for group in grouped_indices]


def dominant_retrieved_course(
    passages: list[dict[str, Any]],
    top_n: int = DOMINANT_COURSE_TOP_N,
) -> str | None:
    if not passages:
        return None

    weighted_scores: dict[str, float] = {}
    for rank, passage in enumerate(passages[:top_n], start=1):
        weight = max(float(passage["score"]), 0.0) + (0.01 * (top_n - rank + 1))
        weighted_scores[passage["course_name"]] = weighted_scores.get(passage["course_name"], 0.0) + weight

    if not weighted_scores:
        return None
    return max(weighted_scores.items(), key=lambda item: item[1])[0]


def inferred_course_alignment(
    passages: list[dict[str, Any]],
    inferred_course_key: str | None,
    top_n: int = DOMINANT_COURSE_TOP_N,
) -> float:
    if not passages or not inferred_course_key:
        return 0.0

    matched_weight = 0.0
    total_weight = 0.0
    for passage in passages[:top_n]:
        weight = max(float(passage["score"]), 0.0) + 0.01
        total_weight += weight
        if entity_matches_course_key(passage, inferred_course_key):
            matched_weight += weight

    if total_weight <= 0:
        return 0.0
    return matched_weight / total_weight


def top_passage_score_total(
    passages: list[dict[str, Any]],
    top_n: int = DOMINANT_COURSE_TOP_N,
) -> float:
    return sum(max(float(passage["score"]), 0.0) for passage in passages[:top_n])


def should_use_inferred_course_fallback(
    first_passages: list[dict[str, Any]],
    fallback_passages: list[dict[str, Any]],
    inferred_course_key: str | None,
) -> bool:
    if not fallback_passages or not inferred_course_key:
        return False

    first_alignment = inferred_course_alignment(first_passages, inferred_course_key)
    fallback_alignment = inferred_course_alignment(fallback_passages, inferred_course_key)
    if fallback_alignment <= first_alignment:
        return False

    first_total = top_passage_score_total(first_passages)
    fallback_total = top_passage_score_total(fallback_passages)
    if fallback_total < first_total * FALLBACK_SCORE_RATIO_THRESHOLD:
        return False

    first_dominant = dominant_retrieved_course(first_passages)
    fallback_dominant = dominant_retrieved_course(fallback_passages)
    if fallback_dominant and entity_matches_course_key(
        {
            "course_name": fallback_dominant,
            "source_relpath": fallback_dominant,
            "source_path": fallback_dominant,
        },
        inferred_course_key,
    ):
        if not first_dominant or not entity_matches_course_key(
            {
                "course_name": first_dominant,
                "source_relpath": first_dominant,
                "source_path": first_dominant,
            },
            inferred_course_key,
        ):
            return True

    return fallback_alignment >= first_alignment + 0.2


def build_passages_from_candidates(
    index: IndexData,
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
    top_k: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if not candidates:
        return [], None

    ordered_by_source, positions = build_source_orders(index)
    support_scores: dict[int, float] = {}
    lecture_scores: dict[str, float] = {}
    preferred_course = pick_preferred_course(
        summarize_group_strength(candidates, "course_name", top_n=4),
        profile,
    )

    for candidate in candidates:
        raw_index = candidate["raw_index"]
        chunk = candidate["chunk"]
        lecture_scores[candidate["lecture_key"]] = max(
            lecture_scores.get(candidate["lecture_key"], 0.0),
            float(candidate["score"]),
        )
        ordered_indices = ordered_by_source[chunk["source_relpath"]]
        center_position = positions[raw_index]
        start = max(0, center_position - NEIGHBOR_WINDOW)
        end = min(len(ordered_indices), center_position + NEIGHBOR_WINDOW + 1)

        for position in range(start, end):
            neighbor_index = ordered_indices[position]
            neighbor_chunk = index.chunks[neighbor_index]
            if neighbor_index != raw_index and not chunks_are_close(chunk, neighbor_chunk):
                continue

            multiplier = 1.0 if neighbor_index == raw_index else 0.92
            support_scores[neighbor_index] = max(
                support_scores.get(neighbor_index, 0.0),
                float(candidate["score"]) * multiplier,
            )

    passages = merge_context_passages(index, support_scores)
    for passage in passages:
        passage["score"] += 0.03 * lecture_scores.get(passage["lecture_key"], 0.0)
        passage["score"] += passage_relevance_adjustment(passage, profile)
        passage["concept_priority"] = passage_concept_priority(passage, profile)

    if profile.get("target_concept_key"):
        passages.sort(
            key=lambda passage: (
                float(passage.get("concept_priority", 0.0)),
                float(passage["score"]),
            ),
            reverse=True,
        )
    else:
        passages.sort(key=lambda passage: passage["score"], reverse=True)
    return passages[:top_k], preferred_course


def retrieve_passages(
    question: str,
    index: IndexData,
    top_k: int,
    min_score: float,
    course_filters: list[str],
    folder_filters: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = build_query_profile(question, course_filters=course_filters, folder_filters=folder_filters)
    candidates = retrieve_candidate_chunks(
        question=question,
        index=index,
        top_k=top_k,
        min_score=min_score,
        course_filters=course_filters,
        folder_filters=folder_filters,
    )
    debug_info = {
        "inferred_course_key": profile["inferred_course_key"],
        "inferred_course_score": float(profile["inferred_course_score"]),
        "inferred_course_confidence": profile["inferred_course_confidence"],
        "dominant_course_first_pass": None,
        "fallback_retrieval_used": False,
        "final_dominant_course": None,
        "preferred_course": None,
    }
    if not candidates:
        return [], debug_info

    passages, preferred_course = build_passages_from_candidates(
        index=index,
        candidates=candidates,
        profile=profile,
        top_k=top_k,
    )
    debug_info["preferred_course"] = preferred_course
    debug_info["dominant_course_first_pass"] = dominant_retrieved_course(passages)

    inferred_course_key = profile["inferred_course_key"]
    inferred_confidence = profile["inferred_course_confidence"]
    first_dominant_course = debug_info["dominant_course_first_pass"]
    should_try_fallback = (
        not profile["course_filters"]
        and inferred_course_key is not None
        and inferred_confidence in {"medium", "high"}
        and first_dominant_course is not None
        and not entity_matches_course_key(
            {
                "course_name": first_dominant_course,
                "source_relpath": first_dominant_course,
                "source_path": first_dominant_course,
            },
            inferred_course_key,
        )
    )

    if should_try_fallback:
        inferred_course_filter = [inferred_course_key]
        fallback_profile = build_query_profile(
            question,
            course_filters=inferred_course_filter,
            folder_filters=folder_filters,
        )
        fallback_candidates = retrieve_candidate_chunks(
            question=question,
            index=index,
            top_k=top_k,
            min_score=min_score,
            course_filters=inferred_course_filter,
            folder_filters=folder_filters,
        )
        fallback_passages, fallback_preferred_course = build_passages_from_candidates(
            index=index,
            candidates=fallback_candidates,
            profile=fallback_profile,
            top_k=top_k,
        )
        if should_use_inferred_course_fallback(passages, fallback_passages, inferred_course_key):
            passages = fallback_passages
            preferred_course = fallback_preferred_course
            debug_info["preferred_course"] = preferred_course
            debug_info["fallback_retrieval_used"] = True

    debug_info["final_dominant_course"] = dominant_retrieved_course(passages)
    return passages, debug_info


def build_context(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for index_number, item in enumerate(results, start=1):
        header_parts = [f"[{index_number}] {item['source_relpath']}"]
        header_parts.append(f"course: {item['course_name']}")
        page_label = format_page_range(item.get("page_start"), item.get("page_end"))
        if page_label:
            header_parts.append(page_label)
        if item.get("titles"):
            header_parts.append("titles: " + "; ".join(item["titles"][:3]))
        parts.append(f"{' | '.join(header_parts)}\n{item['text']}")

    return "\n\n".join(parts)


def build_user_prompt(
    question: str,
    results: list[dict[str, Any]],
    mode: str,
    study_task: str = DEFAULT_STUDY_TASK,
    beginner_mode: bool = False,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    concept_profile = infer_concept_profile(question)
    target_concept_key = concept_profile["target_concept_key"]
    asks_for_calculation = question_requests_calculation(question)
    prompt_results = results
    if study_task == "ask":
        prompt_results = results[: min(3, len(results))]

    context = build_context(prompt_results)
    history_context = format_conversation_history(conversation_history)
    primary_course = results[0]["course_name"] if results else None
    shared_prompt = (
        "Use the retrieved study-material context below as the primary grounding.\n"
        "Answer like a good professor teaching a master's student: clear, complete, well-structured, natural, and faithful to the course material.\n"
        "Write enough to explain the concept properly; do not be artificially brief if the topic needs a fuller explanation.\n"
        f"Exact target request: {question}\n"
        "Strict topic lock: answer that exact concept only.\n"
        "The user has already asked a specific question. Answer it directly; do not ask what the user wants to do with the material.\n"
        "Do not substitute a neighboring concept, broader method, application, or analogy.\n"
        "Do not reinterpret the question.\n"
        "When multiple passages are retrieved, prioritize passages that define or derive the target concept directly over application examples that merely use it.\n"
        "Do not summarize the documents or say things like 'this document appears to...' or 'the slides discuss...'.\n"
        "Do not describe the source material as a document review. Teach the concept directly.\n"
        "First identify the relevant concepts from the materials, then synthesize them into a complete explanation.\n"
        "Avoid awkward slide-fragment phrasing and avoid repeating nearly identical points.\n"
        "If a formula is relevant, explain in plain language what it means instead of just restating symbols.\n"
        "Use local citations like [1] or [2] only for statements grounded in the retrieved course material.\n\n"
    )
    if primary_course:
        shared_prompt += (
            f"Primary retrieved course context: {primary_course}.\n"
            "Stay in that course domain unless another domain is directly relevant to the same concept in the retrieved material.\n\n"
        )

    if study_task == "summarize-lecture":
        task_prompt = (
            "Task type: summarize lecture or topic.\n"
            "Produce a clean study summary with these sections when helpful: Overview, Main concepts, Important formulas or methods, What to remember.\n"
            "Explain how the ideas connect instead of listing isolated bullet points from slides.\n"
        )
    elif study_task == "generate-exam-questions":
        task_prompt = (
            "Task type: generate exam questions.\n"
            "Create 5 to 8 exam-style questions about the exact requested topic.\n"
            "After each question, provide a short expected-answer guide grounded in the retrieved material.\n"
            "Vary the questions across definition, explanation, interpretation, derivation, and application when the context supports that.\n"
        )
    else:
        task_prompt = (
            "Task type: explain concept.\n"
            "Start with a direct definition of the asked concept in one or two sentences.\n"
            "For theoretical concepts, structure the explanation as: definition first, then core components, then assumptions or conditions, then result or implication, then a short intuitive explanation if useful.\n"
            "Do not stop at listing components; explain the theoretical meaning of the concept.\n"
            "For questions about the Gauss-Markov model, connect the functional model and stochastic model to the model's theoretical meaning and BLUE-type result when the course context supports that link.\n"
        )

    audience_prompt = ""
    if beginner_mode:
        audience_prompt = (
            "Beginner mode is enabled.\n"
            "Use simpler wording, define technical terms briefly, and add short intuition where it helps.\n"
            "Assume the user is learning the topic for the first time, but keep the explanation accurate.\n"
        )

    if mode == "strict":
        mode_prompt = (
            "Use only the retrieved local material.\n"
            "If the exact term is missing but the concept can be reconstructed from related concepts in the retrieved material, explain it anyway.\n"
            "Do not use external or general knowledge.\n"
            "If the retrieved material is incomplete, stay on the same concept and reconstruct it from the available local concepts rather than switching topics.\n"
        )
    else:
        mode_prompt = (
            "Prioritize the local material throughout.\n"
            "If the materials are partial, you may fill conceptual gaps with careful standard knowledge as secondary support.\n"
            "Clearly distinguish course-grounded content from added explanatory context.\n"
            "Signal secondary general-knowledge additions briefly with phrases such as \"Based on the course materials...\", \"More generally...\", or \"In standard terminology...\".\n"
            "Do not attach local citations to those secondary additions, and never present them as if they came from the files.\n"
            "If the exact term is not explicitly written but can be reconstructed from the materials plus standard terminology, explain it confidently.\n"
            "If only weak direct explanation exists, still answer the same concept and you may briefly say: \"The materials provide limited direct explanation, but based on them and standard theory...\".\n"
            "Never replace the asked concept with a different one.\n"
        )

    concept_prompt = ""
    if asks_for_calculation:
        concept_prompt += (
            "The user also asks how the concept is calculated.\n"
            "Include the main calculation formula or computation workflow, explain what each matrix, derivative, variance, or covariance term means, and show the simplified special case if the context supports it.\n"
        )

    if target_concept_key == "error propagation":
        concept_prompt += (
            "Target concept family detected: propagation of observation errors / error propagation.\n"
            "In the Adjustment Theory context, explain it as the transfer of uncertainty from the observation vector into derived quantities through the functional model.\n"
            "When the retrieved material supports it, explain the calculation in this order: "
            "set up the functional model, linearize if needed, form the Jacobian or design matrix, combine it with the stochastic model of the observations, then compute the propagated variance-covariance matrix.\n"
            "If supported by the materials, include both the general matrix form Σ_ff = F Σ_ll F^T and the special uncorrelated scalar form based on squared partial derivatives.\n"
            "Prefer direct definition and propagation-law passages over traverse-specific application passages when explaining the concept itself.\n"
            "Do not answer by describing the notes or saying the material is about surveying in general; give the actual concept explanation.\n"
            "Do not drift into traverse closure, misclosure adjustment, or generic measurement-error discussion unless it directly clarifies error propagation itself.\n"
        )

    return (
        shared_prompt
        + task_prompt
        + audience_prompt
        + mode_prompt
        + concept_prompt
        + history_context
        + "Now answer the user's question directly in your own words.\n"
        + f"Question:\n{question}\n\n"
        + f"Context:\n{context}\n"
    )


def format_retrieved_chunks(results: list[dict[str, Any]], retrieval_debug: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Inferred course key: {retrieval_debug.get('inferred_course_key') or '(none)'}")
    lines.append(f"Inferred course score: {float(retrieval_debug.get('inferred_course_score', 0.0)):.3f}")
    lines.append(
        f"Inferred course confidence: {retrieval_debug.get('inferred_course_confidence') or 'low'}"
    )
    lines.append(
        f"Dominant course in first pass: {retrieval_debug.get('dominant_course_first_pass') or '(none)'}"
    )
    lines.append(
        "Fallback retrieval used: "
        + ("yes" if retrieval_debug.get("fallback_retrieval_used") else "no")
    )
    lines.append(f"Final dominant course: {retrieval_debug.get('final_dominant_course') or '(none)'}")

    if not results:
        lines.append("Retrieved chunks used:")
        lines.append("(none)")
        return "\n".join(lines)

    lines.append("Retrieved chunks used:")
    for index_number, item in enumerate(results, start=1):
        header_parts = [
            f"[{index_number}] score={item['score']:.3f}",
            f"course={item['course_name']}",
            f"path={item['source_relpath']}",
        ]
        page_label = format_page_range(item.get("page_start"), item.get("page_end"))
        if page_label:
            header_parts.append(page_label)
        if item.get("titles"):
            header_parts.append("titles=" + "; ".join(item["titles"][:3]))

        lines.append(" | ".join(header_parts))
        lines.append(f"file_path={item['source_path']}")
        lines.append(item["text"])
        lines.append("")

    return "\n".join(lines).rstrip()


def answer_looks_like_meta_summary(answer: str) -> bool:
    normalized = normalize_match_text(answer)
    if not normalized:
        return True

    meta_markers = (
        "the provided text",
        "the provided document",
        "the provided material",
        "this document appears",
        "these slides",
        "the text focuses on",
        "here is a summary",
        "summary of key concepts",
        "if you have a specific question",
        "please provide it",
    )
    return any(marker in normalized for marker in meta_markers)


def build_error_propagation_fallback_answer(
    question: str,
    beginner_mode: bool = False,
) -> str:
    asks_for_calculation = question_requests_calculation(question)
    if beginner_mode:
        intro = (
            "Based on the course materials, propagation of observation errors means: "
            "if your observations are uncertain, that uncertainty is transferred to every quantity you compute from them [1][2]."
        )
        components = (
            "To describe that transfer, the course uses two parts [1]: "
            "the functional model, which tells you how the wanted quantity is computed from the observations, "
            "and the stochastic model, which tells you how uncertain the observations are through their variances and covariances."
        )
        calculation = (
            "To calculate it, you first write the quantity as a function of the observations, for example "
            "x = φ(l). If the relation is nonlinear, you linearize it and build the Jacobian or design matrix F from the partial derivatives [1][2]. "
            "Then you propagate the observation covariance matrix with the general formula Σ_ff = F Σ_ll F^T [1]. "
            "In plain language: F measures how sensitive the result is to each observation, Σ_ll contains the observation uncertainties, "
            "and the product gives the uncertainty of the derived result. "
            "For one derived quantity with uncorrelated observations, this becomes the familiar sum of squared partial-derivative terms, "
            "σ_x² ≈ Σ (∂φ/∂l_i)² σ²_{l_i} [1][2]."
        )
        closing = (
            "So the idea is simple: large observation uncertainty, or a strong sensitivity of the result to one observation, produces a larger propagated error. "
            "If observations are correlated, you must also include covariance terms, or equivalently use the full matrix formula [1]."
        )
    else:
        intro = (
            "Based on the course materials, propagation of observation errors is the transfer of uncertainty from the observation vector "
            "to derived quantities through the functional model [1][2]."
        )
        components = (
            "Its core components are the functional model, which expresses the target quantity as a function of the observations, "
            "and the stochastic model, which describes the variances and covariances of those observations [1]."
        )
        calculation = (
            "For the calculation, you write the quantity as x = φ(l) or, in vector form, f = Φ(l). "
            "If the model is nonlinear, it is linearized, and the Jacobian/design matrix F is formed from the partial derivatives [1][2]. "
            "The propagated variance-covariance matrix is then computed with the general law "
            "Σ_ff = F Σ_ll F^T [1]. "
            "This means: the sensitivity matrix F maps the observation uncertainty Σ_ll into the uncertainty of the derived quantities. "
            "For one derived quantity with uncorrelated observations, the same idea reduces to the special scalar form "
            "σ_x² ≈ Σ (∂φ/∂l_i)² σ²_{l_i}; if observations are correlated, covariance terms must be included, "
            "or equivalently the full matrix formula must be used [1][2]."
        )
        closing = (
            "The theoretical meaning is that error propagation does not correct observations; it quantifies how uncertain the computed result is after the observations "
            "have been combined by the model. That is why it is central both for directly computed quantities and for precision analysis after adjustment [1][2]."
        )

    sections = ["Definition", intro, "Core components", components]
    if asks_for_calculation:
        sections.extend(["How it is calculated", calculation])
    else:
        sections.extend(["Result / implication", closing])
        return "\n\n".join(sections)

    sections.extend(["Short intuition", closing])
    return "\n\n".join(sections)


def build_answer_repair_prompts(
    question: str,
    results: list[dict[str, Any]],
    mode: str,
    beginner_mode: bool,
    bad_answer: str,
) -> tuple[str, str]:
    system_prompt = (
        "You are repairing a failed study-assistant answer. "
        "Rewrite it so it directly answers the user's exact question in a clear, natural, academically strong way. "
        "Do not summarize the documents. "
        "Do not say 'the provided text', 'the document appears', 'the slides discuss', or anything similar. "
        "Do not ask the user what they want to do with the material. "
        "Stay on the exact topic asked. "
        "Use the retrieved local materials as the primary source. "
        "If mode is hybrid-professor and the materials are partial, you may use careful standard knowledge as secondary support, but do not present it as if it came from the files. "
        "If the user asks how something is calculated, include the main formula or computation workflow and explain the symbols in plain language. "
        "Use local citations like [1], [2] only for claims grounded in the retrieved local material."
    )
    if beginner_mode:
        system_prompt += " Beginner mode is enabled, so keep the wording simpler without losing technical correctness."
    if mode == "strict":
        system_prompt += " Strict mode is enabled, so use only the retrieved local material."

    user_prompt = (
        "Repair the following bad draft into a proper direct answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Bad draft:\n{bad_answer}\n\n"
        f"Retrieved context:\n{build_context(results[: min(3, len(results))])}\n"
    )
    return system_prompt, user_prompt


def maybe_apply_answer_guardrail(
    answer: str,
    question: str,
    results: list[dict[str, Any]],
    model: str,
    ollama_host: str,
    mode: str,
    beginner_mode: bool,
) -> str:
    if not answer:
        return answer

    concept_profile = infer_concept_profile(question)
    if not answer_looks_like_meta_summary(answer):
        return answer

    if results:
        repair_system_prompt, repair_user_prompt = build_answer_repair_prompts(
            question=question,
            results=results,
            mode=mode,
            beginner_mode=beginner_mode,
            bad_answer=answer,
        )
        repaired_answer = call_ollama(
            model=model,
            ollama_host=ollama_host,
            system_prompt=repair_system_prompt,
            user_prompt=repair_user_prompt,
        )
        if repaired_answer and not answer_looks_like_meta_summary(repaired_answer):
            return repaired_answer

    if concept_profile["target_concept_key"] == "error propagation" and results:
        return build_error_propagation_fallback_answer(question, beginner_mode=beginner_mode)
    return answer


def call_ollama(
    model: str,
    ollama_host: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.1},
    }
    request = Request(
        f"{ollama_host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {ollama_host}. Make sure 'ollama serve' is running."
        ) from exc

    try:
        return body["message"]["content"].strip()
    except KeyError as exc:
        raise RuntimeError(f"Unexpected Ollama response: {body}") from exc


def format_sources(results: list[dict[str, Any]]) -> str:
    grouped: OrderedDict[str, list[str | None]] = OrderedDict()

    for item in results:
        relpath = item["source_relpath"]
        grouped.setdefault(relpath, [])
        page_label = format_page_range(item.get("page_start"), item.get("page_end"))
        if page_label and page_label not in grouped[relpath]:
            grouped[relpath].append(page_label)

    if not grouped:
        return "No source files were used."

    lines: list[str] = []
    for relpath, page_labels in grouped.items():
        if page_labels:
            lines.append(f"- {relpath} ({', '.join(page_labels)})")
        else:
            lines.append(f"- {relpath}")

    return "\n".join(lines)


def answer_question(
    question: str,
    index: IndexData,
    model: str,
    ollama_host: str,
    mode: str,
    study_task: str,
    beginner_mode: bool,
    top_k: int,
    min_score: float,
    course_filters: list[str],
    folder_filters: list[str],
    show_retrieved: bool,
    conversation_history: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    effective_question = build_effective_question(question, conversation_history)
    results, retrieval_debug = retrieve_passages(
        question=effective_question,
        index=index,
        top_k=top_k,
        min_score=min_score,
        course_filters=course_filters,
        folder_filters=folder_filters,
    )

    if not results:
        if show_retrieved:
            print(format_retrieved_chunks(results, retrieval_debug))
        return "I couldn't find material directly related to that concept in the indexed study materials.", []

    if show_retrieved:
        print(format_retrieved_chunks(results, retrieval_debug))
        print()
    answer = call_ollama(
        model=model,
        ollama_host=ollama_host,
        system_prompt=build_system_prompt(mode, study_task=study_task, beginner_mode=beginner_mode),
        user_prompt=build_user_prompt(
            question,
            results,
            mode,
            study_task=study_task,
            beginner_mode=beginner_mode,
            conversation_history=conversation_history,
        ),
    )
    answer = maybe_apply_answer_guardrail(
        answer,
        question=question,
        results=results,
        model=model,
        ollama_host=ollama_host,
        mode=mode,
        beginner_mode=beginner_mode,
    )
    return answer, results


def run_interactive_session(
    index: IndexData,
    model: str,
    ollama_host: str,
    mode: str,
    study_task: str,
    beginner_mode: bool,
    top_k: int,
    min_score: float,
    course_filters: list[str],
    folder_filters: list[str],
    show_retrieved: bool,
) -> int:
    print("Local RAG assistant ready. Type your question, or 'exit' to quit.")

    while True:
        try:
            question = input("\nQuestion > ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0

        try:
            answer, results = answer_question(
                question=question,
                index=index,
                model=model,
                ollama_host=ollama_host,
                mode=mode,
                study_task=study_task,
                beginner_mode=beginner_mode,
                top_k=top_k,
                min_score=min_score,
                course_filters=course_filters,
                folder_filters=folder_filters,
                show_retrieved=show_retrieved,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print(f"\nAnswer:\n{answer}\n")
        print("Sources:")
        print(format_sources(results))


def main() -> int:
    args = parse_args()

    try:
        index = load_index(args.storage.resolve())
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.question:
        try:
            answer, results = answer_question(
                question=args.question,
                index=index,
                model=args.model,
                ollama_host=args.ollama_host,
                mode=args.mode,
                study_task=args.task,
                beginner_mode=args.beginner,
                top_k=args.top_k,
                min_score=args.min_score,
                course_filters=args.course,
                folder_filters=args.folder,
                show_retrieved=args.show_retrieved,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print("Answer:")
        print(answer)
        print("\nSources:")
        print(format_sources(results))
        return 0

    return run_interactive_session(
        index=index,
        model=args.model,
        ollama_host=args.ollama_host,
        mode=args.mode,
        study_task=args.task,
        beginner_mode=args.beginner,
        top_k=args.top_k,
        min_score=args.min_score,
        course_filters=args.course,
        folder_filters=args.folder,
        show_retrieved=args.show_retrieved,
    )


if __name__ == "__main__":
    raise SystemExit(main())

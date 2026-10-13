"""Offline eval runner over the benchmark portfolio set.

Retrieves context for each benchmark query with BM25, scores the
reference review against what was retrieved, and assembles a report the
RAG Evaluation workflow can publish.

The reference review is held constant and the retrieval is varied, so
relevance measures whether retrieval surfaced query-matching chunks and
faithfulness measures whether the surfaced context can ground a
known-good review. The pair is a retrieval-quality baseline, not a
measurement of live generator output.
"""

from dataclasses import asdict, dataclass, field

import structlog

from rag.evaluator.benchmark import (
    BenchmarkCase,
    BenchmarkProfile,
    load_benchmark,
    sanitize_chunks,
)
from rag.evaluator.eval_suite import EvalSuite
from rag.retriever.keyword_search import KeywordSearcher

logger = structlog.get_logger()

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaseReport:
    """Scores and retrieval detail for one benchmark case."""

    case_id: str
    profile_id: str
    query: str
    chunks_indexed: int
    chunks_retrieved: int
    retrieved_ids: list[str]
    relevance_score: float
    faithfulness_score: float
    overall_score: float


@dataclass(frozen=True)
class ProfileReport:
    """Aggregate score for one benchmark profile."""

    profile_id: str
    cases: int
    mean_overall: float


@dataclass(frozen=True)
class EvalReport:
    """The full benchmark result, serialized to eval_results.json."""

    schema_version: int
    config: dict
    summary: dict
    profiles: list[ProfileReport] = field(default_factory=list)
    cases: list[CaseReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the report as plain JSON-serializable data."""
        return asdict(self)


def _round4(value: float) -> float:
    """Round a score to four decimal places."""
    return round(float(value), 4)


def _mean(values: list[float]) -> float:
    """Return the mean of values, or 0.0 when there are none."""
    if not values:
        return 0.0
    return _round4(sum(values) / len(values))


class EvalRunner:
    """Score the benchmark set and assemble a report."""

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.0,
        suite: EvalSuite | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            top_k: Maximum chunks to retrieve per query
            min_score: Exclusive BM25 floor; hits at or below are dropped
            suite: Evaluation suite to score with, constructed if omitted
        """
        self.top_k = top_k
        self.min_score = min_score
        self.suite = suite or EvalSuite()

    def index(self, chunks: list[dict]) -> KeywordSearcher | None:
        """Build a BM25 index over a chunk corpus, once.

        Sanitizing and tokenizing the corpus is the expensive part of
        retrieval, and every case against the same profile shares it, so
        callers build the index once per profile and query it per case.

        Args:
            chunks: The profile's chunk corpus

        Returns:
            A searcher over the corpus, or None when no chunk has tokens
        """
        corpus = sanitize_chunks(chunks)
        # BM25 divides by the average document length, so a corpus whose
        # documents all tokenize empty has no average to divide by. A
        # whitespace-only text tokenizes empty, so strip() is the same
        # test as split() without allocating a token list per chunk.
        if not any(chunk["text"].strip() for chunk in corpus):
            return None

        searcher = KeywordSearcher()
        searcher.index(corpus)
        return searcher

    def search(self, searcher: KeywordSearcher | None, query: str) -> list[dict]:
        """Query a built index and keep the hits above the score floor.

        Args:
            searcher: The index from ``index``, or None for an empty corpus
            query: The benchmark query

        Returns:
            Matching chunks above the score floor, best first
        """
        if searcher is None:
            return []
        hits = searcher.search(query, top_k=self.top_k)
        return [hit for hit in hits if hit.get("bm25_score", 0.0) > self.min_score]

    def retrieve(self, chunks: list[dict], query: str) -> list[dict]:
        """Index a corpus and retrieve for one query in a single call.

        Convenience for one-off queries; ``run`` indexes each profile once
        and reuses the searcher across its cases instead.

        Args:
            chunks: The profile's chunk corpus
            query: The benchmark query

        Returns:
            Matching chunks above the score floor, best first
        """
        return self.search(self.index(chunks), query)

    def run_case(
        self,
        profile: BenchmarkProfile,
        case: BenchmarkCase,
        searcher: KeywordSearcher | None = None,
    ) -> CaseReport:
        """Score one benchmark case.

        Args:
            profile: The profile the case queries
            case: The case to score
            searcher: A prebuilt index over ``profile.chunks``; built here
                when omitted, so single-case callers need not manage one

        Returns:
            The case's retrieval detail and scores
        """
        if searcher is None:
            searcher = self.index(profile.chunks)
        hits = self.search(searcher, case.query)
        result = self.suite.run(case.query, hits, case.reference_feedback)

        return CaseReport(
            case_id=case.case_id,
            profile_id=profile.profile_id,
            query=case.query,
            chunks_indexed=len(profile.chunks),
            chunks_retrieved=len(hits),
            retrieved_ids=[str(hit.get("id", "")) for hit in hits],
            relevance_score=_round4(result.relevance_score),
            faithfulness_score=_round4(result.faithfulness_score),
            overall_score=_round4(result.overall_score),
        )

    def run(self, profiles: list[BenchmarkProfile]) -> EvalReport:
        """Score every case across every profile.

        Args:
            profiles: The loaded benchmark profiles

        Returns:
            The assembled report
        """
        case_reports: list[CaseReport] = []
        profile_reports: list[ProfileReport] = []

        for profile in profiles:
            searcher = self.index(profile.chunks)
            scored = [self.run_case(profile, case, searcher) for case in profile.cases]
            case_reports.extend(scored)
            profile_reports.append(
                ProfileReport(
                    profile_id=profile.profile_id,
                    cases=len(scored),
                    mean_overall=_mean([report.overall_score for report in scored]),
                )
            )

        summary = {
            "profiles": len(profile_reports),
            "cases": len(case_reports),
            "mean_relevance": _mean([r.relevance_score for r in case_reports]),
            "mean_faithfulness": _mean([r.faithfulness_score for r in case_reports]),
            "mean_overall": _mean([r.overall_score for r in case_reports]),
        }

        logger.info("eval_runner_complete", **summary)

        return EvalReport(
            schema_version=SCHEMA_VERSION,
            config={
                "retriever": "bm25",
                "top_k": self.top_k,
                "min_score": self.min_score,
                "feedback_source": "reference",
                "chunking": "fixed-profile-sections",
            },
            summary=summary,
            profiles=profile_reports,
            cases=case_reports,
        )


def run_benchmark(top_k: int = 5) -> EvalReport:
    """Load the checked-in benchmark set and score it.

    Args:
        top_k: Maximum chunks to retrieve per query

    Returns:
        The assembled report
    """
    return EvalRunner(top_k=top_k).run(load_benchmark())

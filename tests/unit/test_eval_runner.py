"""Tests for runner.py"""

import json

import pytest

from rag.evaluator.benchmark import BenchmarkCase, BenchmarkProfile
from rag.evaluator.runner import EvalRunner, run_benchmark


def _chunk(chunk_id: str, text: str) -> dict:
    """Build a chunk dict for the runner."""
    return {"id": chunk_id, "text": text, "metadata": {}}


def _filler() -> list[dict]:
    """Background chunks with vocabulary the test queries never use.

    BM25 inverse document frequency is zero when a term appears in every
    document and negative when it appears in most of them, so a corpus of
    one or two chunks cannot produce a positive score for anything. Real
    profile corpora run five to nine chunks; these stand in for the rest
    of a portfolio so retrieval behaves the way it does in practice.
    """
    return [
        _chunk("filler-1", "gardening tips for spring bulbs and shade borders"),
        _chunk("filler-2", "sourdough starter maintenance and hydration ratios"),
        _chunk("filler-3", "trail running shoes reviewed for wet limestone"),
        _chunk("filler-4", "watercolour brushes and cold press paper weights"),
    ]


def _profile(profile_id: str, chunks: list[dict], query: str, feedback: str) -> BenchmarkProfile:
    """Build a one-case benchmark profile."""
    return BenchmarkProfile(
        profile_id=profile_id,
        source_file=f"{profile_id}.json",
        chunks=chunks,
        cases=[
            BenchmarkCase(
                case_id=f"{profile_id}-case",
                profile_id=profile_id,
                query=query,
                reference_feedback=feedback,
            )
        ],
    )


@pytest.mark.unit
class TestEvalRunner:
    """Retrieval, scoring, and report assembly."""

    @pytest.fixture
    def runner(self):
        """A runner with default retrieval settings."""
        return EvalRunner()

    def test_retrieve_empty_corpus_returns_empty_list(self, runner):
        assert runner.retrieve([], "python") == []

    def test_retrieve_all_blank_corpus_returns_empty_list(self, runner):
        corpus = [_chunk("a", ""), _chunk("b", "   ")]
        corpus.append({"id": "c", "text": None, "metadata": {}})
        assert runner.retrieve(corpus, "python") == []

    def test_retrieve_ranks_matching_chunk_first(self, runner):
        corpus = [*_filler(), _chunk("match", "python fastapi backend service")]
        assert runner.retrieve(corpus, "python fastapi")[0]["id"] == "match"

    def test_retrieve_respects_top_k(self):
        corpus = [
            *_filler(),
            _chunk("a", "python fastapi service"),
            _chunk("b", "python fastapi tooling"),
        ]
        assert len(EvalRunner(top_k=1).retrieve(corpus, "python fastapi")) == 1

    def test_retrieve_drops_zero_score_hits(self, runner):
        corpus = [*_filler(), _chunk("match", "python fastapi backend")]
        retrieved = [hit["id"] for hit in runner.retrieve(corpus, "python fastapi")]
        assert retrieved == ["match"]

    def test_run_case_scores_are_in_the_unit_interval(self, runner):
        profile = _profile(
            "dev",
            [*_filler(), _chunk("a", "python fastapi backend service with documentation")],
            "python fastapi backend",
            "This developer builds Python backend services. The documentation is thorough.",
        )
        report = runner.run_case(profile, profile.cases[0])

        assert report.chunks_retrieved > 0
        assert 0.0 <= report.relevance_score <= 1.0
        assert 0.0 <= report.faithfulness_score <= 1.0
        expected = (report.relevance_score + report.faithfulness_score) / 2
        assert abs(report.overall_score - expected) < 1e-3

    def test_run_case_with_no_evidence_scores_zero(self, runner):
        profile = _profile(
            "dev",
            [_chunk("a", "")],
            "python fastapi backend",
            "This developer builds Python backend services.",
        )
        report = runner.run_case(profile, profile.cases[0])

        assert report.chunks_retrieved == 0
        assert report.relevance_score == 0.0
        assert report.faithfulness_score == 0.0
        assert report.overall_score == 0.0

    def test_run_case_is_deterministic(self, runner):
        profile = _profile(
            "dev",
            [_chunk("a", "python fastapi backend service")],
            "python fastapi",
            "This developer builds Python backend services.",
        )
        first = runner.run_case(profile, profile.cases[0])
        second = runner.run_case(profile, profile.cases[0])
        assert first == second

    def test_richer_evidence_outscores_thin_evidence(self, runner):
        query = "python fastapi backend documentation"
        feedback = "This developer builds Python backend services with FastAPI documentation."

        rich = _profile(
            "rich",
            [
                *_filler(),
                _chunk("a", "python fastapi backend services with thorough documentation"),
            ],
            query,
            feedback,
        )
        thin = _profile("thin", [*_filler(), _chunk("a", "python")], query, feedback)

        rich_score = runner.run_case(rich, rich.cases[0]).overall_score
        thin_score = runner.run_case(thin, thin.cases[0]).overall_score
        assert rich_score > thin_score

    def test_run_report_shape(self, runner):
        first = _profile("a", [_chunk("x", "python backend")], "python", "Builds Python services.")
        second = _profile("b", [_chunk("y", "react frontend")], "react", "Builds React apps.")

        report = runner.run([first, second])

        assert report.schema_version == 1
        assert report.summary["profiles"] == 2
        assert report.summary["cases"] == 2
        assert len(report.cases) == 2

    def test_report_to_dict_is_json_serializable(self, runner):
        profile = _profile("a", [_chunk("x", "python backend")], "python", "Builds services.")
        payload = json.loads(json.dumps(runner.run([profile]).to_dict(), indent=2))

        assert set(payload) == {"schema_version", "config", "summary", "profiles", "cases"}

    def test_report_scores_are_rounded_to_four_decimals(self, runner):
        profile = _profile(
            "a",
            [_chunk("x", "python fastapi backend documentation")],
            "python fastapi",
            "Builds Python backend services with documentation.",
        )
        report = runner.run([profile])

        for case in report.cases:
            assert round(case.relevance_score, 4) == case.relevance_score
            assert round(case.faithfulness_score, 4) == case.faithfulness_score
            assert round(case.overall_score, 4) == case.overall_score

    def test_index_returns_none_for_tokenless_corpus(self, runner):
        assert runner.index([]) is None
        assert runner.index([_chunk("a", ""), _chunk("b", "   ")]) is None

    def test_search_with_no_index_returns_empty_list(self, runner):
        assert runner.search(None, "python") == []

    def test_retrieve_matches_index_then_search(self, runner):
        corpus = [*_filler(), _chunk("match", "python fastapi backend service")]
        two_step = runner.search(runner.index(corpus), "python fastapi")
        assert runner.retrieve(corpus, "python fastapi") == two_step

    def test_run_indexes_each_profile_once(self):
        """Every case against a profile shares one index build."""

        class CountingRunner(EvalRunner):
            def __init__(self):
                super().__init__()
                self.index_calls = 0

            def index(self, chunks):
                self.index_calls += 1
                return super().index(chunks)

        chunks = [*_filler(), _chunk("a", "python fastapi backend service")]
        cases = [
            BenchmarkCase(f"case-{i}", "dev", "python fastapi", "Builds Python services.")
            for i in range(3)
        ]
        profile = BenchmarkProfile(
            profile_id="dev", source_file="dev.json", chunks=chunks, cases=cases
        )

        runner = CountingRunner()
        report = runner.run([profile])

        assert report.summary["cases"] == 3
        assert runner.index_calls == 1

    def test_run_benchmark_over_the_checked_in_fixtures(self):
        report = run_benchmark()

        assert report.summary["profiles"] == 3
        assert report.summary["cases"] == 6
        for case in report.cases:
            assert 0.0 <= case.relevance_score <= 1.0
            assert 0.0 <= case.faithfulness_score <= 1.0
            assert 0.0 <= case.overall_score <= 1.0

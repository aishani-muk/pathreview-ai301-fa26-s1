"""Benchmark portfolio loading for the offline eval runner.

Turns the checked-in profile fixtures into the chunk dicts the retriever
and the evaluation suite consume, and pairs them with the reference
reviews the benchmark scores against.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_profiles"
DEFAULT_CASES_PATH = REPO_ROOT / "tests" / "fixtures" / "benchmark" / "benchmark_cases.json"


@dataclass(frozen=True)
class BenchmarkCase:
    """One scored query against one benchmark profile."""

    case_id: str
    profile_id: str
    query: str
    reference_feedback: str


@dataclass(frozen=True)
class BenchmarkProfile:
    """A profile fixture, its chunk corpus, and the cases that query it."""

    profile_id: str
    source_file: str
    chunks: list[dict]
    cases: list[BenchmarkCase]


def sanitize_chunks(chunks: list[dict]) -> list[dict]:
    """Return copies of chunks whose text is always a string.

    The scorers in this package read ``chunk["text"]`` and call string
    operations on it, so a missing key or a null value raises. Fixtures
    mirror real GitHub payloads, where an absent README is null, so the
    coercion belongs here at the corpus boundary rather than in the
    scorers.

    Args:
        chunks: Chunk dicts that may carry a missing or null text value

    Returns:
        New chunk dicts with text coerced to a string, inputs unmodified
    """
    return [{**chunk, "text": str(chunk.get("text") or "")} for chunk in chunks]


def profile_to_chunks(profile: dict) -> list[dict]:
    """Build a deterministic chunk corpus from a profile fixture.

    Emits one overview chunk for the profile plus a summary chunk and a
    readme chunk per repository. Chunking is fixed rather than delegated
    to the ingestion chunkers so the benchmark stays offline and its
    scores move only when retrieval or scoring changes.

    Args:
        profile: A parsed profile fixture

    Returns:
        Sanitized chunk dicts, each with an id, text, and metadata
    """
    username = str(profile.get("github_username") or "unknown")
    repos = profile.get("repos") or []

    languages = sorted({str(repo.get("language")) for repo in repos if repo.get("language")})
    overview = (
        f"GitHub profile {username} with portfolio {profile.get('portfolio_url') or 'none'}. "
        f"{len(repos)} public repositories. Languages: {', '.join(languages) or 'none'}."
    )

    chunks: list[dict] = [
        {
            "id": "profile-overview",
            "text": overview,
            "metadata": {"source_id": username, "repo": None, "section": "overview"},
        }
    ]

    for repo in repos:
        name = str(repo.get("name") or "unnamed")
        topics = ", ".join(str(topic) for topic in repo.get("topics") or [])
        summary = (
            f"Repository {name}. {repo.get('description') or 'No description.'} "
            f"Language: {repo.get('language') or 'unknown'}. "
            f"Topics: {topics or 'none'}. Stars: {repo.get('stargazers_count', 0)}."
        )
        chunks.append(
            {
                "id": f"repo::{name}::summary",
                "text": summary,
                "metadata": {"source_id": username, "repo": name, "section": "summary"},
            }
        )
        chunks.append(
            {
                "id": f"repo::{name}::readme",
                "text": repo.get("readme_content"),
                "metadata": {"source_id": username, "repo": name, "section": "readme"},
            }
        )

    return sanitize_chunks(chunks)


def load_profile(path: Path) -> tuple[str, dict]:
    """Load one profile fixture from disk.

    Args:
        path: Path to a profile fixture

    Returns:
        The profile id and the parsed fixture

    Raises:
        FileNotFoundError: If the fixture does not exist
    """
    if not path.is_file():
        raise FileNotFoundError(f"benchmark profile fixture not found: {path}")
    profile = json.loads(path.read_text(encoding="utf-8"))
    return str(profile.get("github_username") or path.stem), profile


def load_benchmark(
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    cases_path: Path = DEFAULT_CASES_PATH,
) -> list[BenchmarkProfile]:
    """Load the benchmark case spec and every profile it references.

    Args:
        profiles_dir: Directory holding the profile fixtures
        cases_path: Path to the benchmark case spec

    Returns:
        One BenchmarkProfile per referenced fixture, in spec order

    Raises:
        FileNotFoundError: If the spec or a referenced fixture is missing
        ValueError: If a case omits a required field
    """
    if not cases_path.is_file():
        raise FileNotFoundError(f"benchmark case spec not found: {cases_path}")
    spec = json.loads(cases_path.read_text(encoding="utf-8"))

    loaded: dict[str, BenchmarkProfile] = {}
    order: list[str] = []

    for raw in spec.get("cases") or []:
        required = ("case_id", "profile", "query", "reference_feedback")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise ValueError(f"benchmark case {raw.get('case_id', '?')} is missing: {missing}")

        source_file = str(raw["profile"])
        if source_file not in loaded:
            profile_id, profile = load_profile(profiles_dir / source_file)
            loaded[source_file] = BenchmarkProfile(
                profile_id=profile_id,
                source_file=source_file,
                chunks=profile_to_chunks(profile),
                cases=[],
            )
            order.append(source_file)

        entry = loaded[source_file]
        entry.cases.append(
            BenchmarkCase(
                case_id=str(raw["case_id"]),
                profile_id=entry.profile_id,
                query=str(raw["query"]),
                reference_feedback=str(raw["reference_feedback"]),
            )
        )

    logger.info(
        "benchmark_loaded",
        profiles=len(order),
        cases=sum(len(loaded[f].cases) for f in order),
    )
    return [loaded[f] for f in order]

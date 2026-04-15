from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


REQUIRED_COLUMNS = ["Case", "Year", "Topic", "Ruling", "Summary"]
BENCHMARK_QUERY_COLUMNS = ["Query", "query"]
BENCHMARK_EXPECTED_COLUMNS = ["ExpectedCase", "Expected Case", "Case", "expected_case"]


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "with"
}


def tokenize(value: object) -> List[str]:
    return [
        token
        for token in normalize_text(value).split()
        if token and (token.isdigit() or token not in STOPWORDS)
    ]


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Use .xlsx, .xls, or .csv."
        )

    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


@dataclass
class SearchResult:
    score: float
    case: str
    year: str
    topic: str
    ruling: str
    summary: str
    explanation: str


class LegalCaseFinder:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = self._prepare_frame(frame)

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(missing)
            )

        prepared = frame[REQUIRED_COLUMNS].copy()

        for column in ["Case", "Topic", "Ruling", "Summary"]:
            prepared[column] = prepared[column].fillna("").astype(str)

        prepared["Year"] = prepared["Year"].fillna("").astype(str)

        prepared["_case_norm"] = prepared["Case"].map(normalize_text)
        prepared["_topic_norm"] = prepared["Topic"].map(normalize_text)
        prepared["_ruling_norm"] = prepared["Ruling"].map(normalize_text)
        prepared["_summary_norm"] = prepared["Summary"].map(normalize_text)
        prepared["_all_norm"] = (
            prepared["_case_norm"]
            + " "
            + prepared["_topic_norm"]
            + " "
            + prepared["_ruling_norm"]
            + " "
            + prepared["_summary_norm"]
            + " "
            + prepared["Year"].map(normalize_text)
        ).str.strip()

        prepared["_case_tokens"] = prepared["_case_norm"].map(set)
        prepared["_topic_tokens"] = prepared["_topic_norm"].map(set)
        prepared["_ruling_tokens"] = prepared["_ruling_norm"].map(set)
        prepared["_summary_tokens"] = prepared["_summary_norm"].map(set)
        prepared["_all_tokens"] = prepared["_all_norm"].map(set)
        return prepared

    @classmethod
    def from_file(cls, path: str | Path) -> "LegalCaseFinder":
        return cls(load_table(path))

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        query_norm = normalize_text(query)
        query_tokens = tokenize(query)
        query_token_set = set(query_tokens)

        scored_results: List[SearchResult] = []

        for _, row in self.frame.iterrows():
            score, explanation = self._score_row(row, query_norm, query_tokens, query_token_set)
            if score > 0:
                scored_results.append(
                    SearchResult(
                        score=round(score, 2),
                        case=row["Case"],
                        year=row["Year"],
                        topic=row["Topic"],
                        ruling=row["Ruling"],
                        summary=row["Summary"],
                        explanation=explanation,
                    )
                )

        scored_results.sort(key=lambda item: (-item.score, item.case))
        return scored_results[:top_k]

    def search_with_total(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], int]:
        query = (query or "").strip()
        if not query:
            return [], 0

        query_norm = normalize_text(query)
        query_tokens = tokenize(query)
        query_token_set = set(query_tokens)

        scored_results: List[SearchResult] = []

        for _, row in self.frame.iterrows():
            score, explanation = self._score_row(row, query_norm, query_tokens, query_token_set)
            if score > 0:
                scored_results.append(
                    SearchResult(
                        score=round(score, 2),
                        case=row["Case"],
                        year=row["Year"],
                        topic=row["Topic"],
                        ruling=row["Ruling"],
                        summary=row["Summary"],
                        explanation=explanation,
                    )
                )

        scored_results.sort(key=lambda item: (-item.score, item.case))
        total_matches = len(scored_results)
        return scored_results[:top_k], total_matches

    def _score_row(
        self,
        row: pd.Series,
        query_norm: str,
        query_tokens: List[str],
        query_token_set: set[str],
    ) -> Tuple[float, str]:
        score = 0.0
        reasons: List[str] = []

        case_norm = row["_case_norm"]
        topic_norm = row["_topic_norm"]
        ruling_norm = row["_ruling_norm"]
        summary_norm = row["_summary_norm"]
        all_norm = row["_all_norm"]

        case_tokens = row["_case_tokens"]
        topic_tokens = row["_topic_tokens"]
        ruling_tokens = row["_ruling_tokens"]
        summary_tokens = row["_summary_tokens"]
        all_tokens = row["_all_tokens"]

        if query_norm in case_norm:
            score += 24
            reasons.append("full query found in case name")
        if query_norm in topic_norm:
            score += 20
            reasons.append("full query found in topic")
        if query_norm in summary_norm:
            score += 14
            reasons.append("full query found in summary")
        if query_norm in ruling_norm:
            score += 8
            reasons.append("full query found in ruling")

        # Bigram / phrase bonus for short legal phrases.
        if len(query_tokens) >= 2:
            bigrams = [
                " ".join(query_tokens[index:index + 2])
                for index in range(len(query_tokens) - 1)
            ]
            for phrase in bigrams:
                if phrase in topic_norm:
                    score += 6
                    reasons.append(f'phrase match in topic: "{phrase}"')
                if phrase in summary_norm:
                    score += 4
                    reasons.append(f'phrase match in summary: "{phrase}"')
                if phrase in case_norm:
                    score += 5
                    reasons.append(f'phrase match in case name: "{phrase}"')

        exact_matches = 0
        partial_matches = 0

        for token in query_tokens:
            token_score_before = score

            if token in case_tokens:
                score += 8
            elif token in case_norm:
                score += 3

            if token in topic_tokens:
                score += 7
            elif len(token) >= 4 and token in topic_norm:
                score += 3

            if token in summary_tokens:
                score += 4
            elif len(token) >= 4 and token in summary_norm:
                score += 2

            if token in ruling_tokens:
                score += 2

            if token.isdigit() and token == normalize_text(row["Year"]):
                score += 7
                reasons.append(f"year match: {row['Year']}")

            if token in all_tokens:
                exact_matches += 1
            elif len(token) >= 4 and token in all_norm:
                partial_matches += 1

            if score > token_score_before:
                reasons.append(f'token match: "{token}"')

        if query_token_set:
            coverage = exact_matches / len(query_token_set)
            score += coverage * 12
            if coverage > 0:
                reasons.append(f"query token coverage: {exact_matches}/{len(query_token_set)}")

        if exact_matches == len(query_token_set) and query_token_set:
            score += 10
            reasons.append("all query tokens matched somewhere in the record")
        elif exact_matches >= max(1, math.ceil(len(query_token_set) / 2)):
            score += 4
            reasons.append("more than half of query tokens matched")

        if exact_matches == 0 and partial_matches > 0:
            score += partial_matches
            reasons.append("partial keyword matches found")

        explanation = "; ".join(dict.fromkeys(reasons))
        return score, explanation

    def evaluate(self, benchmark_path: str | Path, top_k: int = 3) -> Dict[str, object]:
        benchmark = load_table(benchmark_path)

        query_column = find_column(benchmark.columns, BENCHMARK_QUERY_COLUMNS)
        expected_column = find_column(benchmark.columns, BENCHMARK_EXPECTED_COLUMNS)

        if query_column is None or expected_column is None:
            raise ValueError(
                "Benchmark file must include columns for Query and ExpectedCase."
            )

        details: List[Dict[str, object]] = []
        top1_hits = 0
        topk_hits = 0
        reciprocal_ranks: List[float] = []

        for _, row in benchmark.iterrows():
            query = str(row[query_column]).strip()
            expected = str(row[expected_column]).strip().lower()

            results = self.search(query, top_k=top_k)
            predicted_cases = [result.case for result in results]
            predicted_lower = [case.lower() for case in predicted_cases]

            rank = None
            for index, predicted in enumerate(predicted_lower, start=1):
                if predicted == expected:
                    rank = index
                    break

            if rank == 1:
                top1_hits += 1
            if rank is not None:
                topk_hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            details.append(
                {
                    "Query": query,
                    "ExpectedCase": row[expected_column],
                    "TopPrediction": predicted_cases[0] if predicted_cases else "",
                    f"FoundInTop{top_k}": rank is not None,
                    "Rank": rank if rank is not None else "",
                }
            )

        total = len(details)
        return {
            "total_queries": total,
            "top1_accuracy": round(top1_hits / total, 4) if total else 0.0,
            f"top{top_k}_accuracy": round(topk_hits / total, 4) if total else 0.0,
            "mrr": round(sum(reciprocal_ranks) / total, 4) if total else 0.0,
            "details": pd.DataFrame(details),
        }


def print_results(results: Sequence[SearchResult], total_matches: int) -> None:
    if not results:
        print("No matching cases found.")
        return

    print(f"Case results: {len(results)}/{total_matches}")
    print()

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.case} ({result.year})")
        print(f"   Topic: {result.topic}")
        print(f"   Ruling: {result.ruling}")
        print(f"   Score: {result.score}")
        print(f"   Summary: {result.summary}")
        print(f"   Why it matched: {result.explanation}")
        print()

def interactive_loop(finder: LegalCaseFinder, top_k: int) -> None:
    print("Legal Case Precedent Finder")
    print('Enter a search query such as "free speech" or "free, speech".')
    print('Type "top [number]" such as "top 25" to change the number of results.')
    print('Type "quit" or "exit" to stop.\n')

    current_top_k = top_k

    while True:
        query = input(f"Search query [top {current_top_k}]: ").strip()

        if not query:
            continue

        lowered_query = query.lower()

        if lowered_query in {"quit", "exit"}:
            print("Application Quit.")
            break

        if lowered_query.startswith("top "):
            parts = query.split()

            if len(parts) == 2 and parts[1].isdigit():
                new_top_k = int(parts[1])

                if new_top_k > 0:
                    current_top_k = new_top_k
                    print(f"Number of results changed to {current_top_k}.\n")
                else:
                    print("Please enter a number greater than 0.\n")
            else:
                print('Use the format: top 10\n')

            continue

        results, total_matches = finder.search_with_total(query, top_k=current_top_k)
        print_results(results, total_matches)

def main() -> None:
    parser = argparse.ArgumentParser(description="Search landmark Supreme Court cases.")
    parser.add_argument("dataset", help="Path to the Excel or CSV dataset.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to show.")
    parser.add_argument(
        "--query",
        help="Run one query and print results instead of starting interactive mode.",
    )
    parser.add_argument(
        "--evaluate",
        help="Path to a benchmark CSV/XLSX with Query and ExpectedCase columns.",
    )
    parser.add_argument(
        "--save-evaluation",
        help="Optional path to save detailed evaluation results as CSV.",
    )

    args = parser.parse_args()

    finder = LegalCaseFinder.from_file(args.dataset)

    if args.evaluate:
        metrics = finder.evaluate(args.evaluate, top_k=args.top_k)
        print("Evaluation metrics")
        print(f"Total queries: {metrics['total_queries']}")
        print(f"Top-1 accuracy: {metrics['top1_accuracy']}")
        print(f"Top-{args.top_k} accuracy: {metrics[f'top{args.top_k}_accuracy']}")
        print(f"MRR: {metrics['mrr']}")

        if args.save_evaluation:
            metrics["details"].to_csv(args.save_evaluation, index=False)
            print(f"Saved detailed evaluation to {args.save_evaluation}")
        return

    if args.query:
        results, total_matches = finder.search_with_total(args.query, top_k=args.top_k)
        print_results(results, total_matches)
        return

    interactive_loop(finder, top_k=args.top_k)


if __name__ == "__main__":
    main()

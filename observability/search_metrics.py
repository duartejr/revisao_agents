# src/revisao_agents/observability/search_metrics.py

import mlflow


class SearchQualityMetrics:
    """Calculate and log search quality metrics for MLflow."""

    @staticmethod
    def calculate_search_coverage(new_urls: list[str]) -> int:
        """Return the number of new unique URLs found in a single search.

        Args:
            new_urls: New URLs from the current search, excluding previously found ones.

        Returns:
            Count of new unique URLs.
        """
        return len(new_urls)

    @staticmethod
    def calculate_credit_efficiency_individual(credits_used: float) -> float:
        """Return the Tavily credits spent on a single search query.

        Args:
            credits_used: Credits consumed by this specific search.

        Returns:
            Credits per query (equal to ``credits_used`` for an individual search).
        """
        return credits_used

    @staticmethod
    def calculate_credit_efficiency_aggregated(total_credits: float, total_queries: int) -> float:
        """Return the average Tavily credits spent per query across the entire session.

        Args:
            total_credits: Total credits consumed in all searches during the session.
            total_queries: Total number of search queries executed.

        Returns:
            Average credits per query, or ``0.0`` if no queries have been made.
        """
        if total_queries == 0:
            return 0.0
        return total_credits / total_queries

    @staticmethod
    def calculate_result_reuse(urls_search_history: dict[str, int]) -> float:
        """Return the percentage of URLs that appeared in more than one search.

        Formula::

            reused_count = number of URLs with appearance count > 1
            total_appearances = sum of all appearance counts
            result_reuse_pct = (reused_count / total_appearances) * 100

        Args:
            urls_search_history: Mapping of URL → number of times it appeared across
                searches, as maintained by :meth:`update_urls_search_history`.

        Returns:
            Percentage of reused results in the range ``[0.0, 100.0]``. Returns
            ``0.0`` when no URLs have been tracked.
        """
        if not urls_search_history:
            return 0.0

        total_appearances = sum(urls_search_history.values())
        reused_count = sum(count for count in urls_search_history.values() if count > 1)

        if total_appearances == 0:
            return 0.0

        return round((reused_count / total_appearances) * 100, 2)

    @staticmethod
    def log_all_metrics_to_mlflow(metrics: dict[str, float]) -> None:
        """Log all calculated metrics to the active MLflow run.

        Silently skips logging when no MLflow run is active to avoid
        creating anonymous auto-named runs.

        Args:
            metrics: Mapping of metric name → metric value to log.
        """
        import logging

        if not mlflow.active_run():
            logging.getLogger(__name__).debug(
                "log_all_metrics_to_mlflow: no active run — metrics skipped: %s",
                list(metrics.keys()),
            )
            return
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    @staticmethod
    def update_urls_search_history(
        old_history: dict[str, int], new_urls: list[str]
    ) -> dict[str, int]:
        """Return an updated URL appearance-count mapping after a new search.

        Args:
            old_history: The existing ``urls_search_history`` from state. Pass an
                empty dict for the first search.
            new_urls: URLs returned by the current search. May include URLs already
                present in ``old_history``; must *not* be the full historical
                accumulation.

        Returns:
            A new dict with incremented counts for each URL in ``new_urls``.

        Example:
            >>> old = {"http://a.com": 1, "http://b.com": 1}
            >>> new = ["http://a.com", "http://c.com"]
            >>> SearchQualityMetrics.update_urls_search_history(old, new)
            {"http://a.com": 2, "http://b.com": 1, "http://c.com": 1}
        """
        from collections import defaultdict

        urls_history = defaultdict(int, old_history or {})
        for url in new_urls:
            urls_history[url] += 1
        return dict(urls_history)

    @staticmethod
    def calculate_all_search_metrics(
        new_urls: list[str],
        total_accumulated: list[str],
        urls_search_history: dict[str, int],
        credits_used: float,
        total_credits_used: float,
        total_search_queries: int,
    ) -> dict[str, float]:
        """Calculate all four search quality metrics and return them as a dict.

        Convenience method intended for use in graph nodes to avoid repeated
        per-metric calls before logging to MLflow.

        Args:
            new_urls: New URLs found in the current search.
            total_accumulated: All URLs accumulated so far (reserved for future metrics).
            urls_search_history: URL appearance-count mapping maintained across searches.
            credits_used: Tavily credits spent in the current search.
            total_credits_used: Cumulative Tavily credits spent in the session.
            total_search_queries: Total number of search queries in the session.

        Returns:
            dict[str, float]: A flat mapping with the following keys:
                - ``"search_coverage"``: count of new unique URLs this search.
                - ``"result_reuse_percent"``: percentage of URLs seen more than once.
                - ``"credit_efficiency_individual"``: credits spent this search.
                - ``"credit_efficiency_aggregated"``: average credits per query so far.
        """
        return {
            "search_coverage": SearchQualityMetrics.calculate_search_coverage(new_urls),
            "result_reuse_percent": SearchQualityMetrics.calculate_result_reuse(
                urls_search_history
            ),
            "credit_efficiency_individual": SearchQualityMetrics.calculate_credit_efficiency_individual(
                credits_used
            ),
            "credit_efficiency_aggregated": SearchQualityMetrics.calculate_credit_efficiency_aggregated(
                total_credits_used, total_search_queries
            ),
        }

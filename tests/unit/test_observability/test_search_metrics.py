import pytest
from observability.search_metrics import SearchQualityMetrics


class TestSearchCoverage:
    """Unit tests for Search Coverage metric calculation."""

    def test_coverage_empty(self):
        """Test coverage with no new URLs."""
        result = SearchQualityMetrics.calculate_search_coverage([])
        assert result == 0

    def test_coverage_some_urls(self):
        """Test coverage with some new URLs."""
        new_urls = [
            "http://example.com/article1",
            "http://example.com/article2",
            "http://example.com/article3",
        ]
        result = SearchQualityMetrics.calculate_search_coverage(new_urls)
        assert result == 3


class TestCreditEfficiency:
    """Unit tests for Credit Efficiency metric calculation."""

    def test_efficiency_individual(self):
        """Test efficiency for individual search results."""
        credits = 2.5
        result = SearchQualityMetrics.calculate_credit_efficiency_individual(credits)
        assert result == 2.5

    def test_efficiency_aggregated_single_query(self):
        """Test efficiency for aggregated results of a single query should equal total credits."""
        result = SearchQualityMetrics.calculate_credit_efficiency_aggregated(
            total_credits=2.5, total_queries=1
        )
        assert result == 2.5

    def test_efficiency_aggregated_multiple_queries(self):
        """Test efficiency for aggregated results of multiple queries should be total credits divided by total queries."""
        result = SearchQualityMetrics.calculate_credit_efficiency_aggregated(
            total_credits=5.0, total_queries=2
        )
        assert result == 2.5

    def test_efficiency_aggregated_zero_queries(self):
        """Test efficiency for aggregated results with zero queries should return 0 to avoid division by zero."""
        result = SearchQualityMetrics.calculate_credit_efficiency_aggregated(
            total_credits=5.0, total_queries=0
        )
        assert result == 0


class TestResultReuse:
    """Test result reuse percentage calculation."""

    def test_reuse_empty_history(self):
        """Empty history should return 0.0."""
        result = SearchQualityMetrics.calculate_result_reuse({})
        assert result == 0.0

    def test_reuse_no_duplicates(self):
        """All unique URLs should have 0% reuse."""
        history = {
            "https://url1.com": 1,
            "https://url2.com": 1,
            "https://url3.com": 1,
        }
        result = SearchQualityMetrics.calculate_result_reuse(history)
        assert result == 0.0

    def test_reuse_some_duplicates(self):
        """Test with mixed unique and duplicate URLs.

        Example:
            Query 1: [url1, url2, url3]  → {url1:1, url2:1, url3:1}
            Query 2: [url1, url4, url5]  → {url1:2, url2:1, url3:1, url4:1, url5:1}

            Total appearances: 2+1+1+1+1 = 6
            Reused appearances: 2 (all appearances from URLs with count > 1)
            Result: (2/6)*100 = 33.33%
        """
        history = {
            "https://url1.com": 2,
            "https://url2.com": 1,
            "https://url3.com": 1,
            "https://url4.com": 1,
            "https://url5.com": 1,
        }
        result = SearchQualityMetrics.calculate_result_reuse(history)
        assert result == pytest.approx(33.33, rel=0.01)

    def test_reuse_all_duplicates(self):
        """Test with all URLs appearing multiple times."""
        history = {
            "https://url1.com": 3,
            "https://url2.com": 2,
        }
        # Total: 5 appearances, reused: 3+2 = 5
        # Reuse%: (5/5)*100 = 100%
        result = SearchQualityMetrics.calculate_result_reuse(history)
        assert result == 100.0


class TestUpdateUrlsHistory:
    """Test URL history update method."""

    def test_update_empty_old_history(self):
        """Adding URLs to empty history."""
        old = {}
        new = ["https://url1.com", "https://url2.com"]
        result = SearchQualityMetrics.update_urls_search_history(old, new)

        assert result == {
            "https://url1.com": 1,
            "https://url2.com": 1,
        }

    def test_update_increment_existing(self):
        """Adding duplicate URLs should increment counts."""
        old = {
            "https://url1.com": 1,
            "https://url2.com": 1,
        }
        new = ["https://url1.com", "https://url3.com"]
        result = SearchQualityMetrics.update_urls_search_history(old, new)

        assert result == {
            "https://url1.com": 2,  # Incremented!
            "https://url2.com": 1,
            "https://url3.com": 1,
        }

    def test_update_none_old_history(self):
        """None old history should be treated as empty."""
        old = None
        new = ["https://url1.com"]
        result = SearchQualityMetrics.update_urls_search_history(old, new)

        assert result == {"https://url1.com": 1}


class TestCalculateAllMetrics:
    """Test the helper method that calculates all metrics."""

    def test_all_metrics_initial_search(self):
        """Test calculating all metrics for initial search."""
        result = SearchQualityMetrics.calculate_all_search_metrics(
            new_urls=["https://url1.com", "https://url2.com"],
            total_accumulated=["https://url1.com", "https://url2.com"],
            urls_search_history={"https://url1.com": 1, "https://url2.com": 1},
            credits_used=2.5,
            total_credits_used=2.5,
            total_search_queries=1,
        )

        assert "search_coverage" in result
        assert "result_reuse_percent" in result
        assert "credit_efficiency_individual" in result
        assert "credit_efficiency_aggregated" in result

        assert result["search_coverage"] == 2
        assert result["result_reuse_percent"] == 0.0
        assert result["credit_efficiency_individual"] == 2.5
        assert result["credit_efficiency_aggregated"] == 2.5

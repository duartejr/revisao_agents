"""Unit tests for the observability package — MLflow config and tracking helpers."""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# mlflow_config
# ---------------------------------------------------------------------------


class TestMlflowConfig:
    """Tests for observability.mlflow_config constants and env-var reading."""

    def test_experiments_contains_all_five_keys(self):
        from observability.mlflow_config import EXPERIMENTS

        expected = {
            "planning_academic",
            "planning_technical",
            "writing_academic",
            "writing_technical",
            "review_chat",
        }
        assert set(EXPERIMENTS.keys()) == expected

    def test_exp_constants_match_experiments_keys(self):
        from observability.mlflow_config import (
            EXP_PLANNING_ACADEMIC,
            EXP_PLANNING_TECHNICAL,
            EXP_REVIEW_CHAT,
            EXP_WRITING_ACADEMIC,
            EXP_WRITING_TECHNICAL,
            EXPERIMENTS,
        )

        assert EXP_PLANNING_ACADEMIC in EXPERIMENTS
        assert EXP_PLANNING_TECHNICAL in EXPERIMENTS
        assert EXP_WRITING_ACADEMIC in EXPERIMENTS
        assert EXP_WRITING_TECHNICAL in EXPERIMENTS
        assert EXP_REVIEW_CHAT in EXPERIMENTS

    def test_tracking_uri_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///./custom.db")
        # Re-import to pick up monkeypatched env
        import importlib

        import observability.mlflow_config as cfg

        importlib.reload(cfg)
        assert cfg.MLFLOW_TRACKING_URI == "sqlite:///./custom.db"
        importlib.reload(cfg)  # restore original state after assertion

    def test_tracking_uri_has_default(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        import importlib

        import observability.mlflow_config as cfg

        importlib.reload(cfg)
        assert "mlruns" in cfg.MLFLOW_TRACKING_URI
        importlib.reload(cfg)  # restore original state after assertion


# ---------------------------------------------------------------------------
# initialize_experiments
# ---------------------------------------------------------------------------


class TestInitializeExperiments:
    """Tests for observability.mlflow_tracking.initialize_experiments."""

    def test_sets_tracking_uri(self):
        from observability.mlflow_config import MLFLOW_TRACKING_URI

        with (
            patch("observability.mlflow_tracking.mlflow") as mock_mlflow,
            patch("observability.mlflow_tracking.MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI),
        ):
            from observability import initialize_experiments

            initialize_experiments()

        mock_mlflow.set_tracking_uri.assert_called_once_with(MLFLOW_TRACKING_URI)

    def test_creates_all_five_experiments(self):
        from observability.mlflow_config import EXPERIMENTS

        with patch("observability.mlflow_tracking.mlflow") as mock_mlflow:
            from observability import initialize_experiments

            initialize_experiments()

        set_experiment_calls = [c.args[0] for c in mock_mlflow.set_experiment.call_args_list]
        for exp_name in EXPERIMENTS:
            assert exp_name in set_experiment_calls

    def test_is_idempotent(self):
        """Calling twice must call set_experiment exactly 2 * len(EXPERIMENTS) times total."""
        from observability.mlflow_config import EXPERIMENTS

        with patch("observability.mlflow_tracking.mlflow") as mock_mlflow:
            from observability import initialize_experiments

            initialize_experiments()
            initialize_experiments()

        assert mock_mlflow.set_experiment.call_count == 2 * len(EXPERIMENTS)


# ---------------------------------------------------------------------------
# workflow_run
# ---------------------------------------------------------------------------


class TestWorkflowRun:
    """Tests for observability.mlflow_tracking.workflow_run context manager."""

    def test_sets_experiment_and_starts_run(self):
        mock_active_run = MagicMock()
        mock_active_run.__enter__ = MagicMock(return_value=mock_active_run)
        mock_active_run.__exit__ = MagicMock(return_value=False)

        with patch("observability.mlflow_tracking.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value = mock_active_run

            from observability.mlflow_tracking import workflow_run

            with workflow_run("planning_academic", "academic/test-theme"):
                pass

        mock_mlflow.set_experiment.assert_called_with("planning_academic")
        mock_mlflow.start_run.assert_called_once_with(run_name="academic/test-theme")

    def test_logs_params_when_provided(self):
        mock_active_run = MagicMock()
        mock_active_run.__enter__ = MagicMock(return_value=mock_active_run)
        mock_active_run.__exit__ = MagicMock(return_value=False)

        with patch("observability.mlflow_tracking.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value = mock_active_run

            from observability.mlflow_tracking import workflow_run

            params = {"review_type": "academic", "rounds": 3}
            with workflow_run("planning_academic", "run", params=params):
                pass

        mock_mlflow.log_params.assert_called_once_with(params)

    def test_skips_log_params_when_none(self):
        mock_active_run = MagicMock()
        mock_active_run.__enter__ = MagicMock(return_value=mock_active_run)
        mock_active_run.__exit__ = MagicMock(return_value=False)

        with patch("observability.mlflow_tracking.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value = mock_active_run

            from observability.mlflow_tracking import workflow_run

            with workflow_run("planning_academic", "run", params=None):
                pass

        mock_mlflow.log_params.assert_not_called()

    def test_yields_active_run(self):
        mock_active_run = MagicMock()
        mock_active_run.__enter__ = MagicMock(return_value=mock_active_run)
        mock_active_run.__exit__ = MagicMock(return_value=False)

        with patch("observability.mlflow_tracking.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value = mock_active_run

            from observability.mlflow_tracking import workflow_run

            with workflow_run("planning_academic", "run") as active:
                assert active is mock_active_run

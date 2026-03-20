"""Experiment logging with WandB and CSV support."""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def setup_logging(
    log_dir: str = "./outputs/logs",
    experiment_name: str = "experiment",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure logging to file and console.

    Args:
        log_dir: Directory for log files.
        experiment_name: Name for the log file.
        level: Logging level.
        console: Whether to also log to console.

    Returns:
        Configured root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

    logger.info("Logging initialized: %s", log_file)
    return root_logger


class CSVLogger:
    """Simple CSV logger for training metrics.

    Args:
        output_path: Path for the CSV file.
        fieldnames: Column names.
    """

    def __init__(self, output_path: str, fieldnames: List[str]) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

        # Write header
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, metrics: Dict[str, Any]) -> None:
        """Append a row of metrics.

        Args:
            metrics: Dictionary of metric values.
        """
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            row = {k: metrics.get(k, "") for k in self.fieldnames}
            writer.writerow(row)


class WandBLogger:
    """WandB experiment logger wrapper.

    Args:
        project: WandB project name.
        experiment_name: Run name.
        config: Experiment configuration dict.
        entity: WandB entity (username/org).
        enabled: Whether WandB logging is enabled.
    """

    def __init__(
        self,
        project: str = "cross-habitat-marine",
        experiment_name: str = "experiment",
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.run = None

        if not enabled:
            logger.info("WandB logging disabled")
            return

        try:
            import wandb

            self.run = wandb.init(
                project=project,
                name=experiment_name,
                config=config or {},
                entity=entity,
                reinit=True,
            )
            logger.info("WandB initialized: project=%s, run=%s", project, experiment_name)
        except ImportError:
            logger.warning("wandb not installed. Disabling WandB logging.")
            self.enabled = False
        except Exception as e:
            logger.warning("WandB init failed: %s. Disabling.", e)
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metrics.
            step: Optional step number.
        """
        if not self.enabled or self.run is None:
            return

        import wandb

        wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        """Log a file as a WandB artifact.

        Args:
            path: File path.
            name: Artifact name.
            artifact_type: Artifact type.
        """
        if not self.enabled or self.run is None:
            return

        import wandb

        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the WandB run."""
        if self.enabled and self.run is not None:
            import wandb

            wandb.finish()

"""CO2 emissions estimation for experiment tracking.

Supports both the codecarbon library and manual GPU-hour estimation.
Reports per-experiment and total emissions for the paper's efficiency section.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CarbonTracker:
    """Track computational carbon emissions during experiments.

    Uses codecarbon if available, falls back to manual estimation
    based on GPU power consumption and runtime.

    Args:
        project_name: Name of the project for tracking.
        output_dir: Directory for emissions reports.
        country_iso_code: ISO code for carbon intensity (default: USA).
        gpu_power_watts: GPU TDP in watts (default: 300W for V100).
    """

    # Approximate carbon intensity (kgCO2/kWh) by region
    CARBON_INTENSITY = {
        "USA": 0.42,
        "EU": 0.30,
        "CHN": 0.58,
        "global": 0.47,
    }

    def __init__(
        self,
        project_name: str = "cross-habitat-marine",
        output_dir: str = "./outputs/emissions",
        country_iso_code: str = "USA",
        gpu_power_watts: float = 300.0,
    ) -> None:
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.country_iso_code = country_iso_code
        self.gpu_power_watts = gpu_power_watts

        self._tracker = None
        self._start_time: Optional[float] = None
        self._use_codecarbon = False

        # Try to use codecarbon
        try:
            from codecarbon import EmissionsTracker

            self._tracker = EmissionsTracker(
                project_name=project_name,
                output_dir=str(self.output_dir),
                country_iso_code=country_iso_code,
                log_level="warning",
            )
            self._use_codecarbon = True
            logger.info("CarbonTracker initialized with codecarbon")
        except ImportError:
            logger.info(
                "codecarbon not installed. Using manual GPU-hour estimation "
                "(install with: pip install codecarbon)"
            )

    def start(self) -> None:
        """Start tracking emissions."""
        self._start_time = time.time()

        if self._use_codecarbon and self._tracker is not None:
            self._tracker.start()

        logger.info("Carbon tracking started")

    def stop(self) -> Dict[str, float]:
        """Stop tracking and return emissions estimate.

        Returns:
            Dictionary with emissions metrics:
                - gpu_hours: Total GPU hours
                - kwh: Energy consumption in kWh
                - co2_kg: CO2 equivalent in kg
                - duration_seconds: Runtime in seconds
        """
        if self._start_time is None:
            logger.warning("Carbon tracking was not started")
            return {}

        duration = time.time() - self._start_time
        self._start_time = None

        if self._use_codecarbon and self._tracker is not None:
            emissions = self._tracker.stop()
            result = {
                "duration_seconds": duration,
                "gpu_hours": duration / 3600,
                "co2_kg": float(emissions) if emissions else self._estimate_manual(duration)["co2_kg"],
                "kwh": float(emissions) / self.CARBON_INTENSITY.get(self.country_iso_code, 0.42)
                if emissions
                else self._estimate_manual(duration)["kwh"],
                "source": "codecarbon",
            }
        else:
            result = self._estimate_manual(duration)

        logger.info(
            "Carbon tracking stopped: %.1f GPU-hours, %.3f kWh, %.4f kgCO2",
            result["gpu_hours"],
            result["kwh"],
            result["co2_kg"],
        )

        return result

    def _estimate_manual(self, duration_seconds: float) -> Dict[str, float]:
        """Manual emission estimation based on GPU power and runtime.

        Args:
            duration_seconds: Runtime in seconds.

        Returns:
            Estimated emissions.
        """
        gpu_hours = duration_seconds / 3600

        # Energy: power (kW) * time (h)
        kwh = (self.gpu_power_watts / 1000) * gpu_hours

        # PUE (Power Usage Effectiveness) - typical data center ~1.1
        pue = 1.1
        kwh_total = kwh * pue

        # CO2 emissions
        carbon_intensity = self.CARBON_INTENSITY.get(self.country_iso_code, 0.42)
        co2_kg = kwh_total * carbon_intensity

        return {
            "duration_seconds": duration_seconds,
            "gpu_hours": gpu_hours,
            "kwh": kwh_total,
            "co2_kg": co2_kg,
            "pue": pue,
            "carbon_intensity_kgCO2_per_kwh": carbon_intensity,
            "gpu_power_watts": self.gpu_power_watts,
            "source": "manual_estimate",
        }

    def report(self, experiment_emissions: Dict[str, Dict[str, float]]) -> str:
        """Generate a summary report of emissions across experiments.

        Args:
            experiment_emissions: Dict mapping experiment name -> emissions dict.

        Returns:
            Formatted report string.
        """
        lines = ["=" * 50, "Carbon Emissions Report", "=" * 50, ""]

        total_hours = 0
        total_kwh = 0
        total_co2 = 0

        for name, emissions in sorted(experiment_emissions.items()):
            hours = emissions.get("gpu_hours", 0)
            kwh = emissions.get("kwh", 0)
            co2 = emissions.get("co2_kg", 0)

            total_hours += hours
            total_kwh += kwh
            total_co2 += co2

            lines.append(f"  {name}:")
            lines.append(f"    GPU-hours: {hours:.2f}")
            lines.append(f"    Energy:    {kwh:.3f} kWh")
            lines.append(f"    CO2:       {co2:.4f} kgCO2eq")
            lines.append("")

        lines.extend(
            [
                "-" * 50,
                "TOTAL:",
                f"  GPU-hours:  {total_hours:.2f}",
                f"  Energy:     {total_kwh:.3f} kWh",
                f"  CO2:        {total_co2:.4f} kgCO2eq",
                f"  Equivalent: {total_co2 / 0.21:.1f} km driven (avg car)",
                "=" * 50,
            ]
        )

        report = "\n".join(lines)

        # Save to file
        report_path = self.output_dir / "emissions_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info("Emissions report saved: %s", report_path)
        return report

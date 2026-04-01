"""
Data monitoring and automated retraining system.
This can be run as a separate service to monitor data changes.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from src.pipelines.preprocessing_pipeline import run_preprocessing_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.common import read_yaml
from src.utils.logger import setup_logger


class DataMonitor:
    """Monitors data changes and triggers retraining"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.cfg = read_yaml(config_path)
        self.logger = setup_logger()
        self.last_modified = None
        self.is_training = False

    def check_data_changes(self) -> bool:
        """Check if raw data has been modified"""
        raw_data_path = Path(self.cfg["paths"]["raw_data"])

        if not raw_data_path.exists():
            return False

        current_modified = raw_data_path.stat().st_mtime

        if self.last_modified is None:
            self.last_modified = current_modified
            return False

        if current_modified > self.last_modified:
            self.last_modified = current_modified
            return True

        return False

    def retrain_model(self):
        """Retrain the model"""
        if self.is_training:
            self.logger.warning("Training already in progress, skipping...")
            return

        try:
            self.is_training = True
            self.logger.info("🔄 Starting automated retraining...")

            # Run preprocessing
            self.logger.info("📊 Running preprocessing pipeline...")
            preprocess_result = run_preprocessing_pipeline(self.config_path)

            # Run training
            self.logger.info("🤖 Running training pipeline...")
            training_result = run_training_pipeline(self.config_path)

            self.logger.info("✅ Retraining completed successfully!")

        except Exception as e:
            self.logger.error(f"❌ Retraining failed: {str(e)}")
        finally:
            self.is_training = False

    async def monitor_loop(self, check_interval: int = 300):
        """Main monitoring loop"""
        self.logger.info(f"🚀 Starting data monitoring (check every {check_interval}s)...")

        while True:
            try:
                if self.check_data_changes():
                    self.logger.info("📁 Data changes detected!")
                    self.retrain_model()

                await asyncio.sleep(check_interval)

            except Exception as e:
                self.logger.error(f"💥 Monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    def run_sync_monitor(self, check_interval: int = 300):
        """Synchronous monitoring for simple use cases"""
        self.logger.info(f"🚀 Starting synchronous data monitoring (check every {check_interval}s)...")

        while True:
            try:
                if self.check_data_changes():
                    self.logger.info("📁 Data changes detected!")
                    self.retrain_model()

                time.sleep(check_interval)

            except KeyboardInterrupt:
                self.logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"💥 Monitoring error: {str(e)}")
                time.sleep(60)


async def main():
    """Main function for async monitoring"""
    import argparse

    parser = argparse.ArgumentParser(description="Data Monitor for Student Score System")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--interval", type=int, default=300,
                       help="Check interval in seconds (default: 300)")
    parser.add_argument("--sync", action="store_true",
                       help="Use synchronous monitoring instead of async")

    args = parser.parse_args()

    monitor = DataMonitor(args.config)

    if args.sync:
        monitor.run_sync_monitor(args.interval)
    else:
        await monitor.monitor_loop(args.interval)


if __name__ == "__main__":
    # Run async monitoring by default
    asyncio.run(main())
"""Timing and memory tracking utilities."""

import time
import logging
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

import psutil
import numpy as np


class Timer:
    """Timer class for tracking execution time and memory usage."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.memory_usage: Dict[str, Dict[str, float]] = {}
        self._start_times: Dict[str, float] = {}
    
    def start(self, stage: str) -> None:
        """Start timing a stage.

        Args:
            stage: Name of the processing stage
        """
        self._start_times[stage] = time.time()
        self._record_memory(stage, "start")
    
    def stop(self, stage: str) -> float:
        """Stop timing a stage and record duration.

        Args:
            stage: Name of the processing stage

        Returns:
            Duration in seconds
        """
        if stage not in self._start_times:
            raise ValueError(f"Timer for stage '{stage}' was never started")
        
        duration = time.time() - self._start_times[stage]
        self.timings[stage] = duration
        self._record_memory(stage, "end")
        
        return duration
    
    def _record_memory(self, stage: str, point: str) -> None:
        """Record current memory usage.

        Args:
            stage: Name of the processing stage
            point: Point in the stage (start/end)
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        if stage not in self.memory_usage:
            self.memory_usage[stage] = {}
        
        self.memory_usage[stage][point] = {
            'rss': memory_info.rss / (1024 ** 3),  # GB
            'vms': memory_info.vms / (1024 ** 3),  # GB
        }
    
    def report(self) -> None:
        """Report timing and memory usage statistics."""
        if not self.timings:
            logging.warning("No timing data available")
            return
        
        # Calculate total time
        total_time = sum(self.timings.values())
        
        # Print timing report
        logging.info("\nTiming Report:")
        logging.info("-" * 60)
        logging.info(f"{'Stage':<30} {'Duration':<15} {'Percentage':<15}")
        logging.info("-" * 60)
        
        for stage, duration in self.timings.items():
            percentage = (duration / total_time) * 100
            logging.info(
                f"{stage:<30} {duration:>6.2f}s"
                f"{percentage:>14.1f}%"
            )
        
        logging.info("-" * 60)
        logging.info(f"{'Total':<30} {total_time:>6.2f}s")
        
        # Print memory report
        logging.info("\nMemory Usage Report (GB):")
        logging.info("-" * 80)
        logging.info(
            f"{'Stage':<30} {'Start RSS':<12} {'End RSS':<12}"
            f" {'Start VMS':<12} {'End VMS':<12}"
        )
        logging.info("-" * 80)
        
        for stage, points in self.memory_usage.items():
            start = points.get('start', {'rss': 0, 'vms': 0})
            end = points.get('end', {'rss': 0, 'vms': 0})
            
            logging.info(
                f"{stage:<30}"
                f" {start['rss']:>10.2f}  {end['rss']:>10.2f}"
                f" {start['vms']:>10.2f}  {end['vms']:>10.2f}"
            )
        
        logging.info("-" * 80)
    
    def get_summary(self) -> str:
        """Get a string summary of timing and memory usage.

        Returns:
            Summary string
        """
        if not self.timings:
            return "No timing data available"
        
        total_time = sum(self.timings.values())
        lines = []
        
        # Add timing summary
        lines.append("Timing Summary:")
        lines.append("-" * 60)
        lines.append(f"{'Stage':<30} {'Duration':<15} {'Percentage':<15}")
        lines.append("-" * 60)
        
        for stage, duration in self.timings.items():
            percentage = (duration / total_time) * 100
            lines.append(
                f"{stage:<30} {duration:>6.2f}s"
                f"{percentage:>14.1f}%"
            )
        
        lines.append("-" * 60)
        lines.append(f"{'Total':<30} {total_time:>6.2f}s")
        
        # Add memory summary
        lines.append("\nMemory Usage Summary (GB):")
        lines.append("-" * 80)
        lines.append(
            f"{'Stage':<30} {'Peak RSS':<12} {'Peak VMS':<12}"
            f" {'RSS Change':<12} {'VMS Change':<12}"
        )
        lines.append("-" * 80)
        
        for stage, points in self.memory_usage.items():
            start = points.get('start', {'rss': 0, 'vms': 0})
            end = points.get('end', {'rss': 0, 'vms': 0})
            rss_change = end['rss'] - start['rss']
            vms_change = end['vms'] - start['vms']
            lines.append(
                f"{stage:<30} {max(start['rss'], end['rss']):>10.2f}  "
                f"{max(start['vms'], end['vms']):>10.2f}  "
                f"{rss_change:>10.2f}  {vms_change:>10.2f}"
            )
        
        return "\n".join(lines)


@contextmanager
def timed_stage(timer: Timer, stage: str):
    """Context manager for timing a processing stage.

    Args:
        timer: Timer instance
        stage: Name of the processing stage
    """
    timer.start(stage)
    try:
        yield
    finally:
        duration = timer.stop(stage)
        logging.info(f"Completed {stage} in {duration:.2f}s")


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging.debug(
                f"Function '{func.__name__}' completed in {duration:.2f}s"
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(
                f"Function '{func.__name__}' failed after {duration:.2f}s: {str(e)}"
            )
            raise
    
    return wrapper

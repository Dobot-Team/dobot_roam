"""Shared metadata keys emitted by datasets and consumed by trainers."""


class DataMetricKeys:
    """Per-batch data health counters that must not be passed to the model."""

    RESAMPLE_COUNT = "_data_resample_count"
    READ_ATTEMPT_COUNT = "_data_read_attempt_count"
    LOAD_TIME_S = "_data_load_time_s"
    PROCESS_TIME_S = "_data_process_time_s"

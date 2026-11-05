"""Stable task identities shared by data, model and metrics."""
TASK_NAMES = ("tracking", "roaming", "intent", "point_nav", "image_nav")
SOURCE_NAMES = ("dobot_tracking", "dobot_roaming", "trackvla_at", "trackvla_dt", "trackvla_stt")
TASK_SOURCES = {
    "tracking": ("dobot_tracking", "trackvla_at", "trackvla_dt", "trackvla_stt"),
    **{task: ("dobot_roaming",) for task in TASK_NAMES[1:]},
}

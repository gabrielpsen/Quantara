from __future__ import annotations

class QuantaraError(Exception):
    pass

class CheckpointError(QuantaraError):
    pass

class ConfigError(QuantaraError, ValueError):
    pass

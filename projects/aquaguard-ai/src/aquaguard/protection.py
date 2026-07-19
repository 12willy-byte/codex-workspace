from typing import Protocol


class ProtectionLevelProvider(Protocol):
    def protection_level(self, camera_id: str) -> str: ...


class AlarmGate(Protocol):
    def check(self, camera_id: str) -> tuple[bool, str | None]: ...


class AllowAllAlarmGate:
    """Explicit compatibility gate for isolated tests and non-production composition."""

    def check(self, camera_id: str) -> tuple[bool, str | None]:
        return True, None


class ValidatedProtectionAlarmGate:
    required_level = "validated_assistive_alerting"

    def __init__(self, provider: ProtectionLevelProvider):
        self.provider = provider

    def check(self, camera_id: str) -> tuple[bool, str | None]:
        level = self.provider.protection_level(camera_id)
        if level == self.required_level:
            return True, None
        return False, f"camera_protection_level:{level}"

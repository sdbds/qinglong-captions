from dataclasses import dataclass


PARTIAL_CAPTION_EXIT_CODE = 3


@dataclass(frozen=True)
class CaptionBatchOutcome:
    persisted: int
    skipped: int
    failed: int

    @property
    def exit_code(self) -> int:
        if not self.failed:
            return 0
        return PARTIAL_CAPTION_EXIT_CODE if self.persisted else 1

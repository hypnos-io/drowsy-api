from typing import TypedDict


class FatigueDetectionInfo(TypedDict, total=True):
    eyes: dict
    head: dict
    mouth: dict


class FatigueStatus(TypedDict, total=True):
    kssScale: int
    detection: FatigueDetectionInfo

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStrategyEnum:
    STANDARD = "standard"
    LINK_INVENT = "link_invent"
    MOLFORMER = "molformer"
    NO_CURRICULUM = "no_curriculum"

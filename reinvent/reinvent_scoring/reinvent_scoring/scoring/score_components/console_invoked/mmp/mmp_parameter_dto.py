from pydantic import BaseModel, Field, conlist

from reinvent_scoring.scoring.score_components.console_invoked.mmp.mmpdb_parameter_dto import MmpdbParameterDTO


class MMPParameterDTO(BaseModel):
    mmp_temporary_directory: str = None
    mmp_reference_molecules: conlist(str, min_length=1)
    value_mapping: dict = {'MMP': 1.0, 'Not MMP': 0.5}
    mmpdb_parameters: MmpdbParameterDTO = Field(default_factory=MmpdbParameterDTO)
    mmp_debug: bool = False
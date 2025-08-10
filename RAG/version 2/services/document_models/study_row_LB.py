from pydantic import BaseModel, Field
from typing import Optional


class UBStudyRow(BaseModel):
    filename: str | None = Field(
        description='The filename of the source PDF (e.g., "tSCS_upper_body.pdf").'
    )
    publication: str | None = Field(
        description='The publication or citation for the study (e.g., "Inanici et al, 2021").'
    )
    electrode_placement_active: str | None = Field(
        description='Placement of the active electrode(s) (e.g., "C3/C4 and C6/C7").'
    )
    electrode_placement_passive: str | None = Field(
        description='Placement of the passive electrode(s) (e.g., "ASIS").'
    )
    amplitude_mA: str | None = Field(
        description='Stimulation amplitude in milliamps (e.g., "40–90 mA").'
    )
    duration_min_per_session: str | None = Field(
        description='Duration of each stimulation session in minutes (e.g., "60 +/- 20").'
    )
    frequency_Hz_burst: str | None = Field(
        description='Stimulation frequency in Hz or range (e.g., "30", "5Hz–30", or "0.2").'
    )

from typing import List, Optional
from pydantic import BaseModel, Field


class PopulationInfo(BaseModel):
    """Demographics, pathology, and enrolment criteria."""
    sample_size: Optional[int] = Field(
        description="Total number of participants (e.g., 10)."
    )
    sex_breakdown: Optional[str] = Field(
        description='Sex counts (e.g., "9 male, 1 female").'
    )
    age_range_years: Optional[str] = Field(
        description='Age range (e.g., "16–65").'
    )
    condition: Optional[str] = Field(
        description='Primary diagnosis or injury classification '
                    '(e.g., "sub-acute motor-incomplete SCI").'
    )
    time_since_injury: Optional[str] = Field(
        description='Time post-injury/onset (e.g., "2–6 months").'
    )
    inclusion_criteria: Optional[str] = Field(
        description='Key inclusion criteria in plain text.'
    )
    exclusion_criteria: Optional[str] = Field(
        description='Key exclusion criteria in plain text.'
    )


class TherapyParameters(BaseModel):
    """Details of the therapeutic intervention."""
    modality: Optional[str] = Field(
        description='Name of therapy (e.g., "tSCS", "TSS + LT").'
    )
    amplitude_mA: Optional[str] = Field(
        description='Stimulation amplitude or range in mA (e.g., "32–100").'
    )
    frequency_Hz: Optional[str] = Field(
        description='Pulse or burst frequency (e.g., "50").'
    )
    pulse_width_us: Optional[str] = Field(
        description='Pulse width in micro- or milliseconds '
                    '(e.g., "300 μs").'
    )
    duration_min_per_session: Optional[int] = Field(
        description='Minutes per stimulation session (e.g., 30).'
    )
    sessions_per_week: Optional[int] = Field(
        description='Number of sessions per week (e.g., 3).'
    )
    electrode_active: Optional[str] = Field(
        description='Active electrode placement (e.g., "T11/T12").'
    )
    electrode_reference: Optional[str] = Field(
        description='Reference / return electrode placement '
                    '(e.g., "umbilicus").'
    )
    concurrent_training: Optional[str] = Field(
        description='Any simultaneous rehabilitation (e.g., '
                    '"body-weight-supported treadmill LT").'
    )


class MeasurementInfo(BaseModel):
    """Each distinct outcome measure collected in the study."""
    variable: Optional[str] = Field(
        description='Name of variable (e.g., "Peak ankle dorsiflexion").'
    )
    instrument: Optional[str] = Field(
        description='Tool or sensor (e.g., "XSENS IMU").'
    )
    units: Optional[str] = Field(
        description='Measurement units (e.g., "degrees", "oscillations/10 s").'
    )
    protocol: Optional[str] = Field(
        description='Brief description of how the measurement was acquired.'
    )


class ResultEntry(BaseModel):
    """Numerical / categorical results, mapped to time-points and groups."""
    outcome: Optional[str] = Field(
        description='Outcome name matching `variable` in `MeasurementInfo`.'
    )
    group: Optional[str] = Field(
        description='Cohort or comparison arm (e.g., "LT + TSS").'
    )
    timepoint: Optional[str] = Field(
        description='When the measurement was taken (e.g., "post-4 weeks").'
    )
    value: Optional[str] = Field(
        description='Result value or change (e.g., "+6.2 °", "Δ-3 osc").'
    )


class StatisticsInfo(BaseModel):
    """Statistical tests and reported effect sizes."""
    tests_used: Optional[str] = Field(
        description='List of statistical tests (e.g., '
                    '"Wilcoxon signed-rank, Spearman ρ").'
    )
    effect_sizes: Optional[str] = Field(
        description='Effect-size metrics and values '
                    '(e.g., "Cohen d = 0.62").'
    )
    significance_level: Optional[str] = Field(
        description='P-value threshold or multiple-comparison correction.'
    )



class LB_DF_StudyRow(BaseModel):
    """Structured representation of a dorsiflexion / tSCS study."""
    # Meta
    filename: Optional[str] = Field(
        description='Local filename or source PDF (e.g., "Hope2023.pdf").'
    )
    publication: Optional[str] = Field(
        description='Formal citation string.'
    )

    # What the article is about
    study_focus: Optional[str] = Field(
        description='Concise statement of the main research question or topic.'
    )

    # Key domains broken out
    population: Optional[PopulationInfo] = Field(
        description='Demographic and clinical profile of participants.'
    )
    therapy: Optional[TherapyParameters] = Field(
        description='Intervention parameters.'
    )
    measurements: Optional[List[MeasurementInfo]] = Field(
        description='List of all outcome measures collected.'
    )
    experimental_conditions: Optional[str] = Field(
        description='Setup / posture / device context (e.g., '
                    '"overground walk, 10 m, no AFO").'
    )

    # Results & statistics
    results: Optional[List[ResultEntry]] = Field(
        description='Extracted numerical or categorical results.'
    )
    statistics: Optional[StatisticsInfo] = Field(
        description='Statistical procedures and effect sizes reported.'
    )

    # Narrative
    discussion_summary: Optional[str] = Field(
        description='Bullet-point summary of main discussion take-aways.'
    )

class LB_DF_MultipleStudies(BaseModel):
    """Container for multiple LB_DF_StudyRow entries."""
    studies: List[LB_DF_StudyRow] = Field(
        description='List of structured study rows.'
    )
    
    def __getitem__(self, item):
        return self.studies[item]
    
    def __len__(self):
        return len(self.studies)
    
    def __iter__(self):
        return iter(self.studies)
import sys
from pathlib import Path
from typing import List, Optional, Literal, Dict, Union, Type, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, validator
from enum import Enum
from datetime import date, datetime

# English Pydantic Models
class BloodTypeEnum(str, Enum):
    A_POS = "A+"
    A_NEG = "A-"
    B_POS = "B+"
    B_NEG = "B-"
    AB_POS = "AB+"
    AB_NEG = "AB-"
    O_POS = "O+"
    O_NEG = "O-"

class ExamTypeEnum(str, Enum):
    GLUCOSE_TEST = "glucose test"
    HBA1C = "glycated hemoglobin"
    LIPID_PROFILE = "lipid profile"
    CREATININE = "creatinine"
    MICROALBUMIN = "microalbumin"

class GlucoseResult(BaseModel):
    value: float = Field(..., description="Glucose level in mg/dL", ge=0)
    unit: str = Field(default="mg/dL")

class HbA1cResult(BaseModel):
    value: float = Field(..., description="Glycated hemoglobin percentage", ge=0, le=100)
    unit: str = Field(default="%")

class LipidProfileResult(BaseModel):
    cholesterol_total: Optional[float] = Field(None, description="Total cholesterol in mg/dL", ge=0)
    cholesterol_hdl: Optional[float] = Field(None, description="HDL cholesterol in mg/dL", ge=0)
    cholesterol_ldl: Optional[float] = Field(None, description="LDL cholesterol in mg/dL", ge=0)
    triglycerides: Optional[float] = Field(None, description="Triglycerides in mg/dL", ge=0)
    unit: str = Field(default="mg/dL")

class CreatinineResult(BaseModel):
    value: float = Field(..., description="Creatinine level in mg/dL", ge=0)
    unit: str = Field(default="mg/dL")

class MicroalbuminResult(BaseModel):
    value: float = Field(..., description="Microalbumin level in mg/L", ge=0)
    unit: str = Field(default="mg/L")

class IntensityEnum(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

class ActivityTypeEnum(str, Enum):
    WALKING = "walking"
    RUNNING = "running"
    CYCLING = "cycling"
    SWIMMING = "swimming"
    OTHER = "other"

class Activity(BaseModel):
    activity_type: Optional[ActivityTypeEnum] = Field(None, description="Type of activity")
    intensity: Optional[Literal["low", "medium", "high"]] = Field(None, description="Intensity level")
    duration: Optional[str] = Field(None, description="Duration", max_length=50)
    frequency: Optional[str] = Field(None, description="Frequency", max_length=50)

class BloodPressure(BaseModel):
    systolic: Optional[int] = Field(None, description="Systolic pressure in mmHg", ge=0, le=300)
    diastolic: Optional[int] = Field(None, description="Diastolic pressure in mmHg", ge=0, le=200)

class Blood(BaseModel):
    glucose_fasting: Optional[float] = Field(None, description="Fasting glucose in mg/dL", ge=0)
    glucose_postprandial: Optional[float] = Field(None, description="Postprandial glucose in mg/dL", ge=0)
    blood_pressure: Optional[BloodPressure] = Field(None, description="Blood pressure")

class Macros(BaseModel):
    proteins: Optional[float] = Field(None, description="Proteins in grams", ge=0)
    fats: Optional[float] = Field(None, description="Fats in grams", ge=0)
    carbs: Optional[float] = Field(None, description="Carbohydrates in grams", ge=0)

class Nutrition(BaseModel):
    diet_calories: Optional[float] = Field(None, description="Calories in kcal", ge=0)
    diet_macros: Optional[Macros] = Field(None, description="Macronutrients")
    diet_description: Optional[str] = Field(None, description="Diet description", max_length=500)

class MedScreeningInputStatus(str, Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    SKIPPED = "skipped"

# Final data models
class ConstantData(BaseModel):
    name: Optional[str] = Field(None, description="Patient's name - optional", max_length=100)
    gender: Literal["male", "female", "other"] = Field(..., description="Patient's gender - required")
    height: int = Field(..., description="Height in cm - required", ge=0, le=250)
    blood_type: Optional[BloodTypeEnum] = Field(None, description="Blood type - optional")
    date_of_birth: date = Field(..., description="Date of birth - required")

    @validator('date_of_birth')
    def date_of_birth_not_future(cls, v):
        today = date.today()
        if v > today:
            raise ValueError('Date of birth cannot be in the future')
        return v

class FluctuatingData(BaseModel):
    blood: Optional[Blood] = Field(None, description="Blood data - optional")
    stress_level: int = Field(..., description="Stress level (1-10) - required", ge=1, le=10)
    activity: Activity = Field(..., description="Activity - required")
    weight: float = Field(..., description="Weight in kg - required", ge=0, le=300)
    nutrition: Optional[Nutrition] = Field(None, description="Nutrition - optional")

class PeriodicData(BaseModel):
    exam_date: Optional[date] = Field(None, description="Exam date - optional")
    exam_type: Optional[ExamTypeEnum] = Field(None, description="Exam type - optional")
    exam_result: Optional[Dict[ExamTypeEnum, Union[GlucoseResult, HbA1cResult, LipidProfileResult, CreatinineResult, MicroalbuminResult]]] = Field(None, description="Exam result - optional")

class Symptom(BaseModel):
    name: Optional[str] = Field(None, description="Symptom name - optional", max_length=100)
    intensity: Optional[IntensityEnum] = Field(None, description="Symptom intensity - optional")
    duration: Optional[str] = Field(None, description="Duration - optional", max_length=50)
    frequency: Optional[str] = Field(None, description="Frequency - optional", max_length=50)
    onset_date: Optional[date] = Field(None, description="Onset date - optional")

# Setup models (for iterative data collection)
class ConstantDataSetup(BaseModel):
    name: Optional[str] = Field(None, description="Patient's name - optional", max_length=100)
    gender: Optional[Literal["male", "female", "other"]] = Field(None, description="Patient's gender - required")
    height: Optional[int] = Field(None, description="Height in cm - required", ge=0, le=250)
    blood_type: Optional[BloodTypeEnum] = Field(None, description="Blood type - optional")
    date_of_birth: Optional[date] = Field(None, description="Date of birth - required")

    @validator('date_of_birth')
    def date_of_birth_not_future(cls, v):
        if v is None:
            return v
        today = date.today()
        if v > today:
            raise ValueError('Date of birth cannot be in the future')
        return v

class FluctuatingDataSetup(BaseModel):
    blood: Optional[Blood] = Field(None, description="Blood data - optional")
    stress_level: Optional[int] = Field(None, description="Stress level (1-10) - required", ge=1, le=10)
    activity: Optional[Activity] = Field(None, description="Activity - required")
    weight: Optional[float] = Field(None, description="Weight in kg - required", ge=0, le=300)
    nutrition: Optional[Nutrition] = Field(None, description="Nutrition - optional")

class PeriodicDataSetup(BaseModel):
    exam_date: Optional[date] = Field(None, description="Exam date - optional")
    exam_type: Optional[ExamTypeEnum] = Field(None, description="Exam type - optional")
    exam_result: Optional[Dict[ExamTypeEnum, Union[GlucoseResult, HbA1cResult, LipidProfileResult, CreatinineResult, MicroalbuminResult]]] = Field(None, description="Exam result - optional")

class SymptomSetup(BaseModel):
    name: Optional[str] = Field(None, description="Symptom name - optional", max_length=100)
    intensity: Optional[IntensityEnum] = Field(None, description="Symptom intensity - optional")
    duration: Optional[str] = Field(None, description="Duration - optional", max_length=50)
    frequency: Optional[str] = Field(None, description="Frequency - optional", max_length=50)
    onset_date: Optional[date] = Field(None, description="Onset date - optional")

# Chain of Thought models
class ThoughtStep(BaseModel):
    thought: str = Field(
        ...,
        description="What I'm thinking, e.g. 'Provided: height=174 cm; missing required: blood_type; missing optional: none' or 'User refuses to provide: blood_type'",
        max_length=500
    )
    action: str = Field(
        ...,
        description="What I'm doing, e.g. 'Save data', 'Ask for required', 'Ask for optional (once)', 'Set skipped'",
        max_length=100
    )
    action_input: str = Field(
        ...,
        description="Data or question, e.g. 'Height: 174 cm' or 'Please provide blood type' or 'Status: skipped due to refusal'",
        max_length=200
    )
    
class MissingInfo(BaseModel):
    field: str = Field(..., description="Missing information, e.g. 'height'", max_length=50)
    question: str = Field(..., description="Question to user, e.g. 'What is your height?'", max_length=200)

# COT analysis models with iterative approach
class ConstantDataAnalysisCOT(BaseModel):
    thoughts: List[ThoughtStep] = Field(..., description="Analysis steps")
    status: MedScreeningInputStatus = Field(..., description="Status: 'incomplete' if missing required data, 'complete' when all collected, 'skipped' if skipped")
    missing_info: Optional[MissingInfo] = Field(None, description="Missing required data or null if status is 'complete'")
    current_info: ConstantDataSetup = Field(..., description="Current state of collected data")

    model_config = ConfigDict(
        extra="forbid",
        json_encoders={date: lambda v: v.isoformat()}
    )

class FluctuatingDataAnalysisCOT(BaseModel):
    thoughts: List[ThoughtStep] = Field(..., description="Analysis steps")
    status: MedScreeningInputStatus = Field(..., description="Status: 'incomplete' if missing required data, 'complete' when all collected, 'skipped' if skipped")
    missing_info: Optional[MissingInfo] = Field(None, description="List of missing required data or null")
    current_info: FluctuatingDataSetup = Field(..., description="Current state of collected data")

    model_config = ConfigDict(
        extra="forbid",
        json_encoders={date: lambda v: v.isoformat()}
    )

class PeriodicDataAnalysisCOT(BaseModel):
    thoughts: List[ThoughtStep] = Field(..., description="Analysis steps")
    status: MedScreeningInputStatus = Field(..., description="Status: 'incomplete' if missing data, 'complete' when data collected, 'skipped' if skipped")
    missing_info: Optional[MissingInfo] = Field(None, description="List of missing data or null")
    current_info: PeriodicDataSetup = Field(..., description="Current state of collected data")

    model_config = ConfigDict(
        extra="forbid",
        json_encoders={date: lambda v: v.isoformat()}
    )

class SymptomAnalysisCOT(BaseModel):
    thoughts: List[ThoughtStep] = Field(..., description="Analysis steps")
    status: MedScreeningInputStatus = Field(..., description="Status: 'incomplete' if missing data, 'complete' when data collected, 'skipped' if skipped")
    missing_info: Optional[MissingInfo] = Field(None, description="List of missing data or null")
    current_info: SymptomSetup = Field(..., description="Current state of collected data")

    model_config = ConfigDict(
        extra="forbid",
        json_encoders={date: lambda v: v.isoformat()}
    )

import sys
from pathlib import Path
from typing import List, Optional, Literal, Dict, Union, Type, Any
from pydantic import BaseModel, Field, ConfigDict, validator
from enum import Enum
from datetime import date, datetime

# Modele Pydantic
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
    GLUCOSE_TEST = "test glukozy"
    HBA1C = "hemoglobina glikowana"
    LIPID_PROFILE = "lipidogram"
    CREATININE = "kreatynina"
    MICROALBUMIN = "mikroalbumina"

class GlucoseResult(BaseModel):
    value: float = Field(..., description="Poziom glukozy w mg/dL", ge=0)
    unit: str = Field(default="mg/dL")

class HbA1cResult(BaseModel):
    value: float = Field(..., description="Procent hemoglobiny glikowanej", ge=0, le=100)
    unit: str = Field(default="%")

class LipidProfileResult(BaseModel):
    cholesterol_total: Optional[float] = Field(None, description="Całkowity cholesterol w mg/dL", ge=0)
    cholesterol_hdl: Optional[float] = Field(None, description="Cholesterol HDL w mg/dL", ge=0)
    cholesterol_ldl: Optional[float] = Field(None, description="Cholesterol LDL w mg/dL", ge=0)
    triglycerides: Optional[float] = Field(None, description="Triglicerydy w mg/dL", ge=0)
    unit: str = Field(default="mg/dL")

class CreatinineResult(BaseModel):
    value: float = Field(..., description="Poziom kreatyniny w mg/dL", ge=0)
    unit: str = Field(default="mg/dL")

class MicroalbuminResult(BaseModel):
    value: float = Field(..., description="Poziom mikroalbuminy w mg/L", ge=0)
    unit: str = Field(default="mg/L")

class IntensityEnum(str, Enum):
    MILD = "łagodne"
    MODERATE = "umiarkowane"
    SEVERE = "ciężkie"

class ActivityTypeEnum(str, Enum):
    WALKING = "spacer"
    RUNNING = "bieganie"
    CYCLING = "jazda na rowerze"
    SWIMMING = "pływanie"
    OTHER = "inne"

class Activity(BaseModel):
    activity_type: Optional[ActivityTypeEnum] = Field(None, description="Rodzaj aktywności")
    intensity: Optional[Literal["niska", "średnia", "wysoka"]] = Field(None, description="Intensywność")
    duration: Optional[str] = Field(None, description="Czas trwania", max_length=50)
    frequency: Optional[str] = Field(None, description="Częstotliwość", max_length=50)

class BloodPressure(BaseModel):
    systolic: Optional[int] = Field(None, description="Ciśnienie skurczowe w mmHg", ge=0, le=300)
    diastolic: Optional[int] = Field(None, description="Ciśnienie rozkurczowe w mmHg", ge=0, le=200)

class Blood(BaseModel):
    glucose_fasting: Optional[float] = Field(None, description="Glukoza na czczo w mg/dL", ge=0)
    glucose_postprandial: Optional[float] = Field(None, description="Glukoza po posiłku w mg/dL", ge=0)
    blood_pressure: Optional[BloodPressure] = Field(None, description="Ciśnienie krwi")

class Macros(BaseModel):
    proteins: Optional[float] = Field(None, description="Białka w gramach", ge=0)
    fats: Optional[float] = Field(None, description="Tłuszcze w gramach", ge=0)
    carbs: Optional[float] = Field(None, description="Węglowodany w gramach", ge=0)

class Nutrition(BaseModel):
    diet_calories: Optional[float] = Field(None, description="Kalorie w kcal", ge=0)
    diet_macros: Optional[Macros] = Field(None, description="Makroskładniki")
    diet_description: Optional[str] = Field(None, description="Opis diety", max_length=500)

class MedScreeningInputStatus(str, Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    SKIPPED = "skipped"

# Główne modele danych
class ConstantData(BaseModel):
    name: Optional[str] = Field(None, description="Imię pacjenta - opcjonalne", max_length=100)
    gender: Literal["M", "F", "Other"] = Field(..., description="Płeć pacjenta - obowiązkowe")
    height: int = Field(..., description="Wzrost w cm - obowiązkowe", ge=0, le=250)  # Zmieniono na int, dodano le=250
    blood_type: Optional[BloodTypeEnum] = Field(None, description="Grupa krwi - opcjonalne")
    date_of_birth: date = Field(..., description="Data urodzenia - obowiązkowe")

    @validator('date_of_birth')
    def date_of_birth_not_future(cls, v):
        today = date.today()
        if v > today:
            raise ValueError('Data urodzenia nie może być w przyszłości')
        return v

class FluctuatingData(BaseModel):
    blood: Optional[Blood] = Field(None, description="Dane krwi - opcjonalne")
    stress_level: int = Field(..., description="Poziom stresu (1-10) - obowiązkowe", ge=1, le=10)
    activity: Activity = Field(..., description="Aktywność - obowiązkowe")
    weight: float = Field(..., description="Waga w kg - obowiązkowe", ge=0, le=300)  # Dodano le=300
    nutrition: Optional[Nutrition] = Field(None, description="Dieta - opcjonalne")

class PeriodicData(BaseModel):
    exam_date: Optional[date] = Field(None, description="Data badania - opcjonalne")
    exam_type: Optional[ExamTypeEnum] = Field(None, description="Typ badania - opcjonalne")
    exam_result: Optional[Dict[ExamTypeEnum, Union[GlucoseResult, HbA1cResult, LipidProfileResult, CreatinineResult, MicroalbuminResult]]] = Field(None, description="Wynik badania - opcjonalne")

class Symptom(BaseModel):
    name: Optional[str] = Field(None, description="Nazwa objawu - opcjonalne", max_length=100)
    intensity: Optional[IntensityEnum] = Field(None, description="Nasilenie objawu - opcjonalne")
    duration: Optional[str] = Field(None, description="Czas trwania - opcjonalne", max_length=50)
    frequency: Optional[str] = Field(None, description="Częstotliwość - opcjonalne", max_length=50)
    onset_date: Optional[date] = Field(None, description="Data wystąpienia - opcjonalne")

# Modele do zbierania danych
class ConstantDataSetup(BaseModel):
    name: Optional[str] = Field(None, description="Imię pacjenta - opcjonalne", max_length=100)
    gender: Optional[Literal["M", "F", "Other"]] = Field(None, description="Płeć pacjenta - obowiązkowe")
    height: Optional[int] = Field(None, description="Wzrost w cm - obowiązkowe", ge=0, le=250)  # Zmieniono na int, dodano le=250
    blood_type: Optional[BloodTypeEnum] = Field(None, description="Grupa krwi - opcjonalne")
    date_of_birth: Optional[date] = Field(None, description="Data urodzenia - obowiązkowe")

    @validator('date_of_birth')
    def date_of_birth_not_future(cls, v):
        if v is None:
            return v
        today = date.today()
        if v > today:
            raise ValueError('Data urodzenia nie może być w przyszłości')
        return v

class FluctuatingDataSetup(BaseModel):
    blood: Optional[Blood] = Field(None, description="Dane krwi - opcjonalne")
    stress_level: Optional[int] = Field(None, description="Poziom stresu (1-10) - obowiązkowe", ge=1, le=10)
    activity: Optional[Activity] = Field(None, description="Aktywność - obowiązkowe")
    weight: Optional[float] = Field(None, description="Waga w kg - obowiązkowe", ge=0, le=300)  # Dodano le=300
    nutrition: Optional[Nutrition] = Field(None, description="Dieta - opcjonalne")

class PeriodicDataSetup(BaseModel):
    exam_date: Optional[date] = Field(None, description="Data badania - opcjonalne")
    exam_type: Optional[ExamTypeEnum] = Field(None, description="Typ badania - opcjonalne")
    exam_result: Optional[Dict[ExamTypeEnum, Union[GlucoseResult, HbA1cResult, LipidProfileResult, CreatinineResult, MicroalbuminResult]]] = Field(None, description="Wynik badania - opcjonalne")

class SymptomSetup(BaseModel):
    name: Optional[str] = Field(None, description="Nazwa objawu - opcjonalne", max_length=100)
    intensity: Optional[IntensityEnum] = Field(None, description="Nasilenie objawu - opcjonalne")
    duration: Optional[str] = Field(None, description="Czas trwania - opcjonalne", max_length=50)
    frequency: Optional[str] = Field(None, description="Częstotliwość - opcjonalne", max_length=50)
    onset_date: Optional[date] = Field(None, description="Data wystąpienia - opcjonalne")

# Modele Chain of Thought (COT)
class ThoughtStep(BaseModel):
    thought: str = Field(
        ...,
        description="Co myślę, np. 'Podano: wzrost=174 cm; brak obowiązkowych: blood_type; brak opcjonalnych: brak' lub 'Użytkownik odmawia podania: blood_type'",
        max_length=500
    )
    action: str = Field(
        ...,
        description="Co robię, np. 'Zapisz dane', 'Zapytaj o obowiązkowe', 'Zapytaj o opcjonalne (raz)', 'Ustaw skipped'",
        max_length=100
    )
    action_input: str = Field(
        ...,
        description="Dane lub pytanie, np. 'Wzrost: 174 cm' lub 'Podaj grupę krwi' lub 'Status: skipped z powodu odmowy'",
        max_length=200
    )
    
class MissingInfo(BaseModel):
    field: str = Field(..., description="Brakująca informacja, np. 'height'", max_length=50)
    question: str = Field(..., description="Pytanie do użytkownika, np. 'Jaki jest twój wzrost?'", max_length=200)

# Modele analizy COT
class ConstantDataAnalysisCOT(BaseModel):
    thoughts: List[ThoughtStep] = Field(..., description="Etapy analizy krok po kroku")
    status: MedScreeningInputStatus = Field(..., description="Status: 'incomplete' jeśli brakuje danych obowiązkowych, 'complete' gdy wszystkie są zebrane, 'skipped' jeśli pominięto")
    missing_info: Optional[MissingInfo] = Field(None, description="Brakująca dana obowiązkowa lub null, jeśli status to 'incomplete'")
    current_info: ConstantDataSetup = Field(..., description="Aktualny stan zebranych danych")

    model_config = ConfigDict(
        extra="forbid",
        json_encoders={date: lambda v: v.isoformat()}
    )

class FluctuatingDataAnalysisCOT(BaseModel):
    thoughts: List[ThoughtStep] = Field(..., description="Etapy analizy krok po kroku")
    status: MedScreeningInputStatus = Field(..., description="Status: 'incomplete' jeśli brakuje danych obowiązkowych, 'complete' gdy wszystkie są zebrane, 'skipped' jeśli pominięto")
    missing_info: Optional[MissingInfo] = Field(None, description="Lista brakujących danych obowiązkowych lub null")
    current_info: FluctuatingDataSetup = Field(..., description="Aktualny stan zebranych danych")

    model_config = ConfigDict(
        extra="forbid",
        json_encoders={date: lambda v: v.isoformat()}
    )

class PeriodicDataAnalysisCOT(BaseModel):
    thoughts: List[ThoughtStep] = Field(..., description="Etapy analizy krok po kroku")
    status: MedScreeningInputStatus = Field(..., description="Status: 'incomplete' jeśli brakuje danych, 'complete' gdy dane są zebrane, 'skipped' jeśli pominięto")
    missing_info: Optional[MissingInfo] = Field(None, description="Lista brakujących danych lub null")
    current_info: PeriodicDataSetup = Field(..., description="Aktualny stan zebranych danych")

    model_config = ConfigDict(
        extra="forbid",
        json_encoders={date: lambda v: v.isoformat()}
    )

class SymptomAnalysisCOT(BaseModel):
    thoughts: List[ThoughtStep] = Field(..., description="Etapy analizy krok po kroku")
    status: MedScreeningInputStatus = Field(..., description="Status: 'incomplete' jeśli brakuje danych, 'complete' gdy dane są zebrane, 'skipped' jeśli pominięto")
    missing_info: Optional[MissingInfo] = Field(None, description="Lista brakujących danych lub null")
    current_info: SymptomSetup = Field(..., description="Aktualny stan zebranych danych")

    model_config = ConfigDict(
        extra="forbid",
        json_encoders={date: lambda v: v.isoformat()}
    )

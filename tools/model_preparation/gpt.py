from __future__ import annotations

from __future__ import annotations

import sys
import asyncio
import os
from pathlib import Path
import openai
from typing import List, Optional, Literal, Dict, Union, Type, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import instructor
from enum import Enum
from datetime import date
from loguru import logger
# Konfiguracja loggera
logger.remove()
if "thought" not in logger._core.levels:
    logger.level("thought", no=35, color="<red>")
if "action" not in logger._core.levels:
    logger.level("action", no=25, color="<blue>")
if "model_response" not in logger._core.levels:
    logger.level("model_response", no=30, color="<yellow>")

def thought(message, *args, **kwargs):
    logger.log("thought", message, *args, **kwargs)

def action(message, *args, **kwargs):
    logger.log("action", message, *args, **kwargs)

logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.thought = thought
logger.action = action


class GptModel:
    """
    Implementacja modelu GPT (np. GPT-4o) z użyciem OpenAI i biblioteki Instructor.
    Obsługuje generowanie odpowiedzi w formacie Pydantic i tekstowym.
    Jest to samodzielny komponent, który nie zależy od modułu 'backend'.
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None, verbose: Optional[bool] = False):
        """Inicjalizuj model GPT z kluczem API, konfiguracją i opcjonalnym debugowaniem."""
        load_dotenv()  # Załaduj zmienne środowiskowe z pliku .env
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Klucz OpenAI API nie został podany ani znaleziony w zmiennej środowiskowej OPENAI_API_KEY.")
        openai.api_key = self.api_key

        if config:
            self.config = config
        else:
            # Domyślna konfiguracja dla użycia samodzielnego
            self.config = {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 1500
            }
        self.verbose = verbose

    async def model_call(self, 
                        user_input: str, 
                        verbose: bool = False,
                        system_prompt: Optional[str] = None,
                        response_model: Optional[Type[BaseModel]] = None,
                        config: Optional[Dict] = None) -> Any:
        """
        Generuje odpowiedź na podstawie podanego wejścia, używając modelu GPT.
        """
        # Użyj konfiguracji instancji, ale pozwól na jej nadpisanie przez konfigurację specyficzną dla wywołania
        call_config = self.config.copy()
        if config:
            call_config.update(config)

        if not system_prompt:
            system_prompt = """Jesteś pomocnym asystentem, który generuje odpowiedzi na podstawie podanego wejścia.
            Używaj poprawnej polszczyzny i zachowaj jasność w odpowiedziach."""

        try:
            # Wyciągnij nazwę modelu z konfiguracji, aby uniknąć duplikatu argumentu
            model_name = call_config.pop("model", "gpt-3.5-turbo")

            if response_model:
                client = instructor.patch(openai, mode=instructor.Mode.MD_JSON)
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    response_model=response_model,
                    **call_config,
                )
                final_output = response
            else:
                response = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    **call_config,
                )
                final_output = response.choices[0].message.content

            if verbose or self.verbose:
                if hasattr(response, 'thoughts'):
                    for t in getattr(response, 'thoughts', []):
                        logger.thought(t)
                logger.model_response(final_output)

            return final_output
        except Exception as e:
            logger.error(f"Błąd podczas wywołania modelu GPT: {e}")
            return None

            raise

# Sekcja testowa
if __name__ == "__main__":
    async def run_tests():
        try:
            gpt = GptModel(verbose=True)
            logger.info("Model GPT zainicjowany pomyślnie dla testów lokalnych.")

            # Przykład z polskimi znakami
            system_prompt = """Otrzymasz opis objawów pacjenta, jego historii choroby i wyników badań oraz model Pydantic ze szczegółowym opisem, co ma się znajdować w odpowiedzi.
            Twoim zadaniem jest stworzenie poprawnego JSON-a ze zwięzłym, spójnym i logicznym opisem klinicznym, który nie ogranicza się do powtarzania informacji, ale wyciąga istotne wnioski.
            Używaj poprawnej polszczyzny i zachowaj jasność w odpowiedziach."""

            user_prompt = """Od jakiegoś czasu czuję się osłabiony, zwłaszcza rano. Mam cukrzycę typu 2 od kilku lat, biorę metforminę, ale ostatnio gorzej się czuję. W wynikach badań ostatnio wyszedł mi podwyższony poziom hemoglobiny glikowanej, a lekarz wspominał coś o możliwości zmiany leczenia. Mam mniej energii i czuję, że coś jest nie tak."""

            class PatientData(BaseModel):
                symptoms: List[str] = Field(..., description="Lista objawów zgłaszanych przez pacjenta")
                medical_history: List[str] = Field(..., description="Historia choroby pacjenta")
                test_results: Optional[Dict[str, str]] = Field(None, description="Wyniki badań pacjenta w formacie klucz-wartość")
                medications: Optional[List[str]] = Field(None, description="Lista aktualnie przyjmowanych leków")
                lifestyle_factors: Optional[List[str]] = Field(None, description="Czynniki stylu życia mogące mieć wpływ na zdrowie pacjenta")

            try:
                result = await gpt.model_call(
                    user_input=user_prompt,
                    verbose=True,
                    system_prompt=system_prompt,
                    response_model=PatientData,
                )
                logger.info(f"Wynik testu z modelem Pydantic:\n{result}")
                print(f"Wynik: {result}")
                print(f"Typ wyniku: {type(result)}")
            except Exception as e:
                logger.error(f"Błąd podczas testowania modelu Pydantic: {e}")
                print(f"Błąd: {e}")

        except Exception as e:
            logger.error(f"Błąd inicjalizacji modelu: {e}")

    # Uruchom testy asynchroniczne
    asyncio.run(run_tests())
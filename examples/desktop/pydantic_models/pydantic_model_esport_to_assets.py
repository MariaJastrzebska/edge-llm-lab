import json
import os
import sys
from pathlib import Path
from pydantic import BaseModel

# Dodaj ścieżkę główną projektu do sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model_preparation.pydantic_models.pydantic_models import (
    ConstantData, 
    Symptom, 
    FluctuatingData, 
    PeriodicData,
    ConstantDataAnalysisCOT,
    FluctuatingDataAnalysisCOT,
    PeriodicDataAnalysisCOT,
    SymptomAnalysisCOT
)

# Import English models
from model_preparation.pydantic_models.pydantic_models_en import (
    ConstantData as ConstantDataEN,
    Symptom as SymptomEN,
    FluctuatingData as FluctuatingDataEN,
    PeriodicData as PeriodicDataEN,
    ConstantDataAnalysisCOT as ConstantDataAnalysisCOTEN,
    FluctuatingDataAnalysisCOT as FluctuatingDataAnalysisCOTEN,
    PeriodicDataAnalysisCOT as PeriodicDataAnalysisCOTEN,
    SymptomAnalysisCOT as SymptomAnalysisCOTEN
)

# Ścieżki do folderów assets
base_assets_path = "/Users/mariamalycha/Documents/fed-mobile/fed_mobile_chat_flutter/assets"
schemas_long_path = os.path.join(base_assets_path, "schemas_long")
schemas_path = os.path.join(base_assets_path, "schemas")

# Upewnij się, że foldery istnieją
os.makedirs(schemas_long_path, exist_ok=True)
os.makedirs(schemas_path, exist_ok=True)

def save_schema(model_class: type[BaseModel], output_path: str):
    """Zapisuje pełny schemat modelu do pliku (oryginalna funkcja)."""
    schema = model_class.model_json_schema()
    schema["model_name"] = model_class.__name__
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    
    print(f"Zapisano pełny schemat {model_class.__name__} do {output_path}")

def min_schema(model_class: type[BaseModel]) -> dict:
    """Generuje minimalistyczny schemat Pydantic: tylko properties, typy, enumy i required."""
    full_schema = model_class.model_json_schema()
    props = {}
    for field_name, field_info in full_schema["properties"].items():
        # Sprawdź, czy pole ma bezpośredni 'type'
        if "type" in field_info:
            props[field_name] = {"type": field_info["type"]}
            if "enum" in field_info:
                props[field_name]["enum"] = field_info["enum"]
        # Obsługa pól z $ref (odwołania do $defs)
        elif "$ref" in field_info:
            ref_key = field_info["$ref"].split("/")[-1]
            if ref_key in full_schema["$defs"]:
                ref_info = full_schema["$defs"][ref_key]
                props[field_name] = {"type": ref_info.get("type", "object")}
                if "enum" in ref_info:
                    props[field_name]["enum"] = ref_info["enum"]
                elif "properties" in ref_info:
                    props[field_name]["properties"] = ref_info["properties"]
        # Obsługa zagnieżdżonych obiektów w items (np. listy)
        elif "items" in field_info and "$ref" in field_info["items"]:
            ref_key = field_info["items"]["$ref"].split("/")[-1]
            if ref_key in full_schema["$defs"]:
                ref_info = full_schema["$defs"][ref_key]
                props[field_name] = {"type": "array", "items": {"type": ref_info.get("type", "object")}}
                if "properties" in ref_info:
                    props[field_name]["items"]["properties"] = ref_info["properties"]
    
    return {
        "properties": props,
        "required": full_schema.get("required", [])
    }

def save_min_schema(model_class: type[BaseModel], output_path: str):
    """Zapisuje minimalistyczny schemat modelu do pliku."""
    schema = min_schema(model_class)
    schema["model_name"] = model_class.__name__
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    
    print(f"Zapisano zoptymalizowany schemat {model_class.__name__} do {output_path}")

def main():
    # Polish models
    selected_models_pl = [
        ConstantData,
        Symptom,
        FluctuatingData,
        PeriodicData,
        ConstantDataAnalysisCOT,
        FluctuatingDataAnalysisCOT,
        PeriodicDataAnalysisCOT,
        SymptomAnalysisCOT
    ]
    
    # English models  
    selected_models_en = [
        ConstantDataEN,
        SymptomEN,
        FluctuatingDataEN,
        PeriodicDataEN,
        ConstantDataAnalysisCOTEN,
        FluctuatingDataAnalysisCOTEN,
        PeriodicDataAnalysisCOTEN,
        SymptomAnalysisCOTEN
    ]

    all_models = [
        ("pl", selected_models_pl),
        ("en", selected_models_en)
    ]

    all_model_names = []

    # Generuj schematy dla obu języków
    for language, models in all_models:
        print(f"\n🌍 Generating schemas for {language.upper()}...")
        
        for model in models:
            model_name = model.__name__
            suffix = f"_{language}" if language == "en" else ""
            
            # Full schema in schemas_long
            full_output_file = os.path.join(schemas_long_path, f"{model_name.lower()}{suffix}_schema.json")
            save_schema(model, full_output_file)
            
            # Minimal schema in schemas
            min_output_file = os.path.join(schemas_path, f"{model_name.lower()}{suffix}_schema.json")
            save_min_schema(model, min_output_file)
            
            all_model_names.append(f"{model_name}{suffix}")

    # Generate index file with all models
    model_index = {
        "models": {
            "polish": [model.__name__ for model in selected_models_pl],
            "english": [f"{model.__name__}_en" for model in selected_models_en]
        },
        "description": "Medical models index (full in schemas_long, optimized in schemas) - Polish and English versions"
    }
    with open(os.path.join(schemas_long_path, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Schema generation completed for both languages!")
    print(f"📊 Generated {len(all_model_names)} schemas total")

if __name__ == "__main__":
    main()
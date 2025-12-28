if __name__ == "__main__":
    from referenced_clean import EvalModelsReferenced
    from unreferenced_clean import EvalModelsUnreferenced
    from base_eval import BaseEvaluation, Agent
    
    # === KONFIGURACJA GLOBALNYCH USTAWIEŃ ===
    print("\n🔍 KONFIGURACJA STEP-BY-STEP EVALUATION")
    print("="*60)
    
    # Użyj stałego agenta
    agent_type_enum = Agent.CONSTANT_DATA_EN
    print(f"🤖 Agent: {agent_type_enum.value}")
    
    # 1. Wybór modeli do testowania
    print("\n📋 WYBÓR MODELI DO TESTOWANIA:")
    print("1. Wszystkie modele z config.yaml")
    print("2. Tylko modele z tested: true bez wyników w logach")
    mode_choice = input("Wybierz tryb: 1 - Wszystkie modele z config, 2 - Tylko tested: true bez wyników: ")
    
    # 2. Czy automatycznie pobierać modele
    print("\n📦 POBIERANIE MODELI:")
    auto_install = input("Czy automatycznie pobierać brakujące modele? (y/n): ").lower().strip()
    install_choice = "y" if auto_install in ["y", "yes", "tak"] else "n"
    
    print("\n" + "="*60)
    print("✅ KONFIGURACJA ZAKOŃCZONA")
    print("="*60)
    
    # Najlepsza optymalizacja z analizy
    best_optimization = [
        {},  # Baseline
        {"--flash-attn": None, "--cont-batching": None},  # Najlepsza kombinacja (-24.5% latencji)
    ]
    
    print(f"🚀 Optymalizacje do testowania:")
    print(f"   1. Baseline (bez optymalizacji)")
    print(f"   2. Flash Attention + Continuous Batching (najlepsza: -24.5% latencji)")
    
    # Pobierz modele do testowania
    if mode_choice == "1":
        models_to_evaluate = EvalModelsReferenced.get_truly_untested_models(agent_type_enum.value, "referenced", only_tested_true=False)
        print("📋 Tryb: Wszystkie modele z config")
    else:
        models_to_evaluate = EvalModelsReferenced.get_truly_untested_models(agent_type_enum.value, "referenced", only_tested_true=True)
        print("📋 Tryb: Tylko tested: true bez wyników")
    
    if not models_to_evaluate:
        print(f"❌ Brak modeli do testowania dla agenta {agent_type_enum.value}")
        exit(1)
        
    total_models = len(models_to_evaluate)
    print(f"📊 Znaleziono {total_models} modeli do testowania")

    # === PĘTLA PRZEZ MODELE ===
    for i, model_name in enumerate(models_to_evaluate, 1):
        print(f"\n{'='*80}")
        print(f"🎯 MODEL {i}/{total_models}: {model_name}")
        print(f"{'='*80}")

        # Sprawdź dostępność modelu (używaj globalnego ustawienia)
        if not BaseEvaluation.check_model_availability(model_name, install_choice=install_choice):
            print(f"⏭️  Pomijam model {model_name}...")
            continue
            
        print(f"✅ Model dostępny: {model_name}")
        print(f"✅ Agent: {agent_type_enum.value}")
        
        # === REFERENCED EVALUATION ===
        print(f"\n🔍 REFERENCED EVALUATION - {model_name}")
        print(f"-" * 60)
        
        try:
            referenced_evaluator = EvalModelsReferenced(
                model_name=model_name,
                agent=agent_type_enum
            )
            
            # Uruchom referenced z najlepszymi optymalizacjami (tylko logi)
            referenced_evaluator.pipeline_eval_model(
                mode="logs_only",  # Tylko logi, bez wizualizacji
                use_cache=True,
                optimisations=best_optimization
            )
            
            print(f"✅ Referenced evaluation zakończona dla {model_name}")
            del referenced_evaluator  # Zwolnij pamięć
            
        except Exception as e:
            print(f"❌ Błąd w referenced evaluation dla {model_name}: {e}")
            continue  # Przejdź do następnego modelu
        
        # === UNREFERENCED EVALUATION ===
        print(f"\n🏥 UNREFERENCED EVALUATION - {model_name}")
        print(f"-" * 60)
        
        try:
            unreferenced_evaluator = EvalModelsUnreferenced(
                model_name=model_name,
                agent=agent_type_enum
            )
            
            # Uruchom unreferenced z najlepszymi optymalizacjami (tylko logi)
            unreferenced_evaluator.pipeline_eval_model(
                mode="logs_only",  # Tylko logi, bez wizualizacji
                use_cache=True,
                optimisations=best_optimization
            )
            
            print(f"✅ Unreferenced evaluation zakończona dla {model_name}")
            del unreferenced_evaluator  # Zwolnij pamięć
            
        except Exception as e:
            print(f"❌ Błąd w unreferenced evaluation dla {model_name}: {e}")
        
        # === OZNACZ MODEL JAKO TESTED ===
        try:
            from model_config_loader import mark_model_as_tested
            mark_model_as_tested(agent_type_enum.value, model_name)
            print(f"💾 Model {model_name} oznaczony jako tested: true")
        except Exception as e:
            print(f"⚠️ Nie udało się oznaczyć modelu jako tested: {e}")
        
        print(f"🎉 Model {model_name} zakończony - referenced + unreferenced")

    # === PODSUMOWANIE ===
    print(f"\n{'='*80}")
    print(f"🏆 EWALUACJA STEP-BY-STEP ZAKOŃCZONA!")
    print(f"{'='*80}")
    print(f"📊 Przetestowano {total_models} modeli")
    print(f"🔧 Każdy model: Referenced → Unreferenced")
    print(f"⚡ Optymalizacje: Baseline + Flash Attention + Continuous Batching")
    print(f"📝 Tryb: Tylko logi (bez wizualizacji)")
    print(f"\n💡 Aby wygenerować wizualizacje, uruchom:")
    print(f"   - referenced_clean.py z mode='viz_only'")
    print(f"   - unreferenced_clean.py z mode='viz_only'")
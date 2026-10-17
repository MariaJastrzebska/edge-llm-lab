"""Extracted evaluator methods; historical plotting contracts are preserved."""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt

class ReferencedMetrics:
    def evaluate_with_gpt_judge(self, llm_response, reference_response, context_messages=None, use_cache=True):
        """
        Używa ChatGPT jako 'judge' do oceny jakości odpowiedzi modelu.
        Zwraca strukturę z oceną (0-1), szczegółami i kryteriami.

        """
        print(f" Evaluating with GPT Judge ")
        context_str = "\n".join(
            [f"{m['role']}: {m['content']}" for m in (context_messages or [])])

        # Check if we're using English agent
        is_english = hasattr(self, 'agent') and self.agent_type.endswith('_en')
        #load prompt
        prompt = self.read_txt(self.EVALUATION_PROMPT_PATH)
        prompt = prompt.format(
            context_str=context_str,
            reference_response=reference_response,
            llm_response=llm_response
        )


        judge_messages = [
            {"role": "system", "content": "You are an objective, rigorous judge of response quality." if is_english else "Jesteś obiektywnym, rygorystycznym sędzią jakości odpowiedzi."},
            {"role": "user", "content": prompt},
        ]
        try:
            # Use structured output with Pydantic model
            judge_response = self.get_openai_response(
                messages=judge_messages,
                custom_catch_type="gpt_judge",
                use_cache=use_cache,
                response_model=self.GPTJudgeResponse
            )

            # Parse JSON string if needed (from cache)
            if isinstance(judge_response, str):
                try:
                    judge_response = json.loads(judge_response)
                except json.JSONDecodeError as e:
                    print(f" Failed to parse GPT Judge response as JSON: {e}")
                    print(f" Response was: {judge_response[:200]}...")
                    raise

            # Extract structured output directly (already in correct format)
            if isinstance(judge_response, dict) and 'arguments' in judge_response:
                result = judge_response['arguments']
            else:
                result = judge_response

            # Calculate average score from all criteria
            criteria_scores = result['criteria_scores']
            all_scores = [v['score'] for v in criteria_scores.values()]
            total = sum(all_scores) / len(all_scores) if all_scores else 0.0

            # Prepare criteria description
            criteria_desc = ', '.join([f'{k}:{v["score"]}' for k, v in criteria_scores.items()])

            if use_cache:
                if not hasattr(self, 'gpt_judge_cache'):
                    self.gpt_judge_cache = {}
                custom_catch = {
                    'timestamp': datetime.now().isoformat(),
                    # Normalizacja do 0-1 dla spójności z innymi metrykami
                    'score': float(total) / 10.0,
                    # Oryginalny wynik 0-10 do wyświetlania
                    'original_score': float(total),
                    'details': f"Judge average score: {total:.1f}/10 (from {len(all_scores)} criteria)",
                    'explanation': f"Average of {len(all_scores)} criteria: {criteria_desc}",
                    'criteria_scores': result['criteria_scores'],
                    'strengths': result.get('overall_strengths', []),
                    'weaknesses': result.get('overall_weaknesses', []),
                }
                # self.save_cache(cache_key, custom_catch = custom_catch, filename="gpt_judge_scores")
            return {
                'score': float(total) / 10.0,  # Normalize to 0-1
                'original_score': float(total),  # Keep 0-10 for display
                'details': f"Judge average score: {total:.1f}/10 (from {len(all_scores)} criteria)",
                'explanation': f"Average of {len(all_scores)} criteria: {criteria_desc}",
                'criteria_scores': result['criteria_scores'],
                'strengths': result.get('overall_strengths', []),
                'weaknesses': result.get('overall_weaknesses', []),
            }
        except Exception as e:
            print(f" Error in GPT Judge evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'score': None, 'status': 'error',
                'details': f'Judge evaluation failed: {type(e).__name__}'
            }


    def calculate_fast_metrics(self, llm_response, reference_response, use_cache):
        """
        Kompletna analiza odpowiedzi LLM vs referencja:
        - Metryki tekstowe (BLEU, ROUGE, Jaccard, Levenshtein, METEOR)
        - Metryki tool call (nazwy funkcji, argumenty, struktura)
        - Metryki poprawności formatu (JSON validity, wymagane pola)
        """
        if getattr(self, 'metrics_profile', 'full') in {'structured', 'text'}:
            from edge_llm_lab.metrics.grading import grade_response
            return grade_response(llm_response, reference_response, self.metrics_profile)
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.translate.meteor_score import single_meteor_score
            from nltk.tokenize import word_tokenize
            from rouge_score import rouge_scorer
            from Levenshtein import distance as levenshtein_distance
            try:
                from bert_score import score as bert_score
                BERT_SCORE_AVAILABLE = True
            except ImportError:
                BERT_SCORE_AVAILABLE = False
            NLTK_AVAILABLE = True
        except ImportError as e:
            NLTK_AVAILABLE = False
            print(
                f"Warning: Missing libraries for metrics ({e}). Skipping advanced metrics.")

        results = {}
        llm_str = llm_response
        ref_str = reference_response

        print(f"--------Response text:---------- \n {llm_str}\n")
        print(f"--------Reference text:---------- \n {ref_str}\n")

        # === NORMALIZACJA JSON DLA METRYK TEKSTOWYCH ===
        # Parsuj i znormalizuj JSONy dla sprawiedliwego porównania tekstowego
        llm_normalized = llm_str
        ref_normalized = ref_str

        try:
            llm_parsed = json.loads(llm_str)
            # Znormalizowany z sort_keys=True
            llm_normalized = self.pretty_json(llm_parsed, 'str')
        except json.JSONDecodeError:
            pass  # Użyj oryginalnego stringa jeśli nie jest JSON

        try:
            ref_parsed = json.loads(ref_str)
            # Znormalizowany z sort_keys=True
            ref_normalized = self.pretty_json(ref_parsed, 'str')
        except json.JSONDecodeError:
            pass  # Użyj oryginalnego stringa jeśli nie jest JSON

        # === 1. METRYKI POPRAWNOŚCI FORMATU ===

        # JSON Validity
        llm_json_valid = False
        ref_json_valid = False
        llm_parsed = None
        ref_parsed = None

        try:
            llm_parsed = json.loads(llm_str)
            llm_json_valid = True
        except json.JSONDecodeError:
            pass

        try:
            ref_parsed = json.loads(ref_str)
            ref_json_valid = True
        except json.JSONDecodeError:
            pass

        results['json_validity'] = {
            'score': 1.0 if llm_json_valid else 0.0,
            'details': f'LLM JSON valid: {llm_json_valid}, Reference JSON valid: {ref_json_valid}'
        }

        # Required Fields Check (dla tool calls)
        required_fields_score = 0.0
        if llm_json_valid and ref_json_valid:
            required_fields = ['tool_calls']
            llm_has_fields = all(
                field in llm_parsed for field in required_fields)
            ref_has_fields = all(
                field in ref_parsed for field in required_fields)
            if llm_has_fields and ref_has_fields:
                required_fields_score = 1.0
            elif llm_has_fields or ref_has_fields:
                required_fields_score = 0.5

        results['required_fields'] = {
            'score': required_fields_score,
            'details': 'Checks if response contains required fields like tool_calls'
        }

        # === 2. METRYKI TOOL CALLS ===

        if llm_json_valid and ref_json_valid and 'tool_calls' in llm_parsed and 'tool_calls' in ref_parsed:
            llm_tools = llm_parsed.get('tool_calls', [])
            ref_tools = ref_parsed.get('tool_calls', [])

            # Tool Call Count Match
            tool_count_match = 1.0 if len(llm_tools) == len(ref_tools) else 0.0
            results['tool_count_match'] = {
                'score': tool_count_match,
                'details': f'LLM tools: {len(llm_tools)}, Reference tools: {len(ref_tools)}'
            }

            # Tool Names Match
            if llm_tools and ref_tools:
                llm_names = [tool.get('name', '') for tool in llm_tools]
                ref_names = [tool.get('name', '') for tool in ref_tools]
                print(f"DEBUG: LLM tool names: {llm_names}")
                print(f"DEBUG: Ref tool names: {ref_names}")

                name_matches = sum(1 for ln, rn in zip(llm_names, ref_names) if ln == rn)
                tool_names_score = name_matches / max(len(llm_names), len(ref_names)) if max(len(llm_names), len(ref_names)) > 0 else 0.0

                print(f"DEBUG: Name matches: {name_matches}, Score: {tool_names_score}")
            else:
                tool_names_score = 1.0 if not llm_tools and not ref_tools else 0.0
                print(f"DEBUG: No tools to compare, score: {tool_names_score}")

            results['tool_names_match'] = {
                'score': tool_names_score,
                'details': f'Tool name matches: {name_matches if llm_tools and ref_tools else "N/A"} (LLM: {len(llm_tools) if llm_tools else 0}, Ref: {len(ref_tools) if ref_tools else 0})'
            }

            # Arguments Structure & Value Match
            if llm_tools and ref_tools:
                arg_struct_matches = 0
                arg_value_matches = 0
                total_comparisons = min(len(llm_tools), len(ref_tools))

                for i in range(total_comparisons):
                    llm_args = llm_tools[i].get('arguments', {})
                    ref_args = ref_tools[i].get('arguments', {})

                    if isinstance(llm_args, dict) and isinstance(ref_args, dict):
                        llm_keys = set(llm_args.keys())
                        ref_keys = set(ref_args.keys())

                        if llm_keys == ref_keys:
                            arg_struct_matches += 1
                            # Check values
                            correct_values = 0
                            for k in llm_keys:
                                if str(llm_args[k]).lower().strip() == str(ref_args[k]).lower().strip():
                                    correct_values += 1
                            if correct_values == len(llm_keys) and len(llm_keys) > 0:
                                arg_value_matches += 1

                args_struct_score = arg_struct_matches / total_comparisons if total_comparisons > 0 else 0.0
                args_value_score = arg_value_matches / total_comparisons if total_comparisons > 0 else 0.0
            else:
                args_struct_score = 1.0 if not llm_tools and not ref_tools else 0.0
                args_value_score = 1.0 if not llm_tools and not ref_tools else 0.0

            results['tool_args_structure'] = {
                'score': args_struct_score,
                'details': f'Argument keys match: {arg_struct_matches if llm_tools and ref_tools else "N/A"}/{total_comparisons if llm_tools and ref_tools else "N/A"}'
            }
            results['tool_args_values'] = {
                'score': args_value_score,
                'details': f'Exact argument value match: {arg_value_matches if llm_tools and ref_tools else "N/A"}/{total_comparisons if llm_tools and ref_tools else "N/A"}'
            }
        else:
            # No valid tool calls to compare
            results['tool_count_match'] = {
                'score': 0.0, 'details': 'Invalid JSON or missing tool_calls'}
            results['tool_names_match'] = {
                'score': 0.0, 'details': 'Invalid JSON or missing tool_calls'}
            results['tool_args_structure'] = {
                'score': 0.0, 'details': 'Invalid JSON or missing tool_calls'}

        # === 3. METRYKI TEKSTOWE (używają znormalizowanych JSONów) ===

        if NLTK_AVAILABLE:
            # Tokenizacja znormalizowanych stringów
            llm_tokens = word_tokenize(llm_normalized)
            ref_tokens = word_tokenize(ref_normalized)

            # Jaccard Similarity
            set1 = set(llm_tokens)
            set2 = set(ref_tokens)
            if not set1 and not set2:
                jaccard_score = 1.0
            else:
                jaccard_score = len(set1.intersection(set2)
                                    ) / len(set1.union(set2))
            results['jaccard_similarity'] = {
                'score': jaccard_score,
                'details': 'Jaccard similarity between token sets (normalized JSON).'
            }

            # Levenshtein Similarity
            max_len = max(len(llm_normalized), len(ref_normalized))
            if max_len > 0:
                lev_dist = levenshtein_distance(llm_normalized, ref_normalized)
                lev_similarity = 1 - (lev_dist / max_len)
            else:
                lev_similarity = 1.0
            results['levenshtein_similarity'] = {
                'score': lev_similarity,
                'details': 'Normalized Levenshtein similarity (normalized JSON).'
            }

            # BLEU Score
            bleu = sentence_bleu([ref_tokens], llm_tokens,
                                 smoothing_function=SmoothingFunction().method1)
            results['bleu'] = {
                'score': bleu,
                'details': 'BLEU score (normalized JSON tokens).'
            }

            # BERTScore
            if BERT_SCORE_AVAILABLE:
                try:
                    # bert_score returns (P, R, F1)
                    P, R, F1 = bert_score([llm_normalized], [ref_normalized], lang='en' if self.agent_type.endswith('_en') else 'pl', verbose=False)
                    results['p_bert'] = {'score': float(P[0]), 'details': 'BERTScore Precision (hallucination detection)'}
                    results['r_bert'] = {'score': float(R[0]), 'details': 'BERTScore Recall (completeness)'}
                except Exception as e:
                    print(f"BERTScore evaluation failed: {e}")

            # ROUGE-L Score
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge = scorer.score(ref_normalized, llm_normalized)[
                'rougeL'].fmeasure
            results['rougeL'] = {
                'score': rouge,
                'details': 'ROUGE-L score for summarization quality (normalized JSON).'
            }

            # METEOR Score
            meteor = single_meteor_score(ref_tokens, llm_tokens)
            results['meteor'] = {
                'score': meteor,
                'details': 'METEOR score with synonym and stemming awareness (normalized JSON).'
            }
        # Integracja ChatGPT Judge (jeśli dostępny klient)
        print(" Evaluating with GPT Judge...")

        # Przekaż tylko kontekst z bieżącej rundy, nie całą historię
        current_round_context = getattr(self, 'current_round_context', None)

        if not getattr(self, 'OPENAI_CLIENT', None):
            results['gpt_judge'] = {'score': None, 'status': 'unavailable'}
            return results
        gpt_judge_result = self.evaluate_with_gpt_judge(
            llm_response,
            reference_response,
            context_messages=current_round_context,
            use_cache=use_cache
        )

        # Dodaj kategorie GPT Judge jako osobne metryki
        if 'criteria_scores' in gpt_judge_result and isinstance(gpt_judge_result['criteria_scores'], dict):
            for criterion, criterion_data in gpt_judge_result['criteria_scores'].items():
                if isinstance(criterion_data, dict) and 'score' in criterion_data:
                    # Nowy format: {"score": X, "explanation": "..."}
                    results[f'gpt_{criterion}'] = {
                        # Normalizacja z 1-10 na 0-1
                        'score': float(criterion_data['score']) / 10.0,
                        'details': f'GPT Judge - {criterion}: {criterion_data["score"]}/10',
                        'explanation': criterion_data.get('explanation', 'Brak wyjaśnienia')
                    }
                else:
                    # Stary format: tylko liczba
                    results[f'gpt_{criterion}'] = {
                        # Normalizacja z 1-10 na 0-1
                        'score': float(criterion_data) / 10.0,
                        'details': f'GPT Judge - {criterion}: {criterion_data}/10'
                    }

        results['gpt_judge'] = gpt_judge_result
        print(
            f" GPT Judge evaluation completed: {(gpt_judge_result.get('original_score') or 0.0):.1f}/10")

        return results

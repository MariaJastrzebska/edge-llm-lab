# import mlflow
# import mlflow.pyfunc
# import subprocess
# import json
# import time
# import os
# import sys
# import argparse
# import pandas as pd
# import torch
# from openai import OpenAI
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer
# from sklearn.metrics import f1_score
# from sentence_transformers import SentenceTransformer, util
# from dotenv import load_dotenv
# import instructor

# # Determine project root and add to sys.path
# # Assumes script is in model_preparation/scripts/evaluation/
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.append(PROJECT_ROOT)

# # Import Pydantic models - ensure this path is correct relative to PROJECT_ROOT
# # Based on notebook: from model_preparation.pydantic_models.pydantic_models import ConstantDataAnalysisCOT
# # For now, only importing what's confirmed used and exists
# from model_preparation.pydantic_models.pydantic_models import ConstantDataAnalysisCOT

# # --- Configuration ---
# load_dotenv() # Load environment variables from .env file

# MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# PROMPTS_DIR = os.path.join(PROJECT_ROOT, "fed_mobile_chat_flutter/assets/prompts")
# # PROMPTS_TEST_DIR = os.path.join(PROJECT_ROOT, "fed_mobile_chat_flutter/assets/ollamatest") # Uncomment if used

# LOCAL_MODELS = [
#     "llama3-groq-tool-use",
#     "hermes3",
#     "granite3.1-dense:2b",
#     "granite3.2:2b",
#     "phi4-mini"
# ]

# # Agents to test - ensure prompt files and test inputs match these agent names
# AGENTS = [
#     "constant_data_agent",
#     # "symptoms_agent", 
#     # "fluctuating_data_agent",
#     # "periodic_agent",
#     # "medical_challenge_agent",
#     # "feedback_agent" # Notebook had 'feedback', ensure consistency or add input
# ]

# # Test inputs for each agent
# TEST_INPUTS = {
#     "constant_data_agent": "Mam na imię Maria, mam 175 cm.",
#     # "symptoms_agent": "Czuję zmęczenie, umiarkowane, od 3 dni.",
#     # "fluctuating_data_agent": "Poziom glukozy 120 mg/dL, waga 71 kg, biegam 30 minut, średnia intensywność.",
#     # "periodic_agent": "Zmęczenie utrzymuje się od 2 tygodni, częstotliwość: codziennie.",
#     # "feedback_agent": "Udało się, trudność średnia, zauważyłam mniejsze zmęczenie, ocena 3.",
#     # "medical_challenge_agent": "Przez 5 dni utrzymuj poziom glukozy poniżej 140 mg/dL.",
# }

# # Pydantic models for structured output with 'instructor'
# # Add other agents and their Pydantic models here as needed
# COT_FUNCTION_CALL_PYDANTIC_MODELS = {
#     "constant_data_agent": ConstantDataAnalysisCOT,
#     # "symptoms_agent": SymptomsAnalysisCOT, 
#     # ... other agents
# }

# # --- MLflow and OpenAI Client Setup ---
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# if OPENAI_API_KEY:
#     openai_client = OpenAI(api_key=OPENAI_API_KEY)
#     # Patch the client for 'instructor'
#     patched_openai_client = instructor.patch(openai_client, mode=instructor.Mode.MD_JSON)
# else:
#     openai_client = None
#     patched_openai_client = None
#     print("Warning: OPENAI_API_KEY not found. ChatGPT tests will be skipped.")

# # --- Utility Functions ---
# def load_prompt(agent_name, prompts_base_dir):
#     # Assumes prompt file is named like 'agent_name_without_agent_suffix.txt'
#     # e.g., for 'constant_data_agent', prompt file is 'constant_data.txt'
#     prompt_filename = agent_name.replace('_agent', '') + ".txt"
#     prompt_file_path = os.path.join(prompts_base_dir, prompt_filename)
#     try:
#         with open(prompt_file_path, "r", encoding="utf-8") as f:
#             return f.read()
#     except FileNotFoundError:
#         print(f"Warning: Prompt file not found for agent {agent_name} at {prompt_file_path}")
#         return None

# # --- Metric Calculation Functions ---
# def calculate_perplexity(text, model_name="gpt2"):
#     try:
#         tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         model = GPT2LMHeadModel.from_pretrained(model_name)
#         inputs = tokenizer(text, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**inputs, labels=inputs["input_ids"])
#         loss = outputs.loss
#         return torch.exp(loss).item()
#     except Exception as e:
#         print(f"Error calculating perplexity: {e}")
#         return float('nan')

# def calculate_bleu(reference, candidate):
#     reference_tokens = [reference.split()]
#     candidate_tokens = candidate.split()
#     return sentence_bleu(reference_tokens, candidate_tokens)

# def calculate_rouge(reference, candidate):
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = scorer.score(reference, candidate)
#     return scores['rougeL'].fmeasure

# def calculate_f1_intent(predicted_intent, true_intent):
#     # This is a placeholder, adapt as per your specific intent definition
#     return f1_score([str(true_intent)], [str(predicted_intent)], average='binary', pos_label=str(true_intent), zero_division=0)

# def calculate_semantic_similarity(reference, candidate):
#     try:
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         embeddings1 = model.encode(reference, convert_to_tensor=True)
#         embeddings2 = model.encode(candidate, convert_to_tensor=True)
#         return util.cos_sim(embeddings1, embeddings2).item()
#     except Exception as e:
#         print(f"Error calculating semantic similarity: {e}")
#         return float('nan')

# def evaluate_technical_metrics(response_str, user_input, agent_name):
#     # This function is highly specific to 'constant_data_agent' as per the notebook.
#     # Generalize or make conditional if evaluating other agents this way.
#     if agent_name != "constant_data_agent":
#         # print(f"Technical metrics evaluation not implemented for agent: {agent_name}")
#         return float('nan') # Or a default score / skip
#     try:
#         data = json.loads(response_str)
#         # The notebook expects a 'tool_calls' like structure for Ollama, 
#         # and direct content for ChatGPT. This needs to be harmonized or handled conditionally.
#         # Assuming 'data' is the direct argument dictionary after instructor parsing for ChatGPT
#         # or the content of 'tool_calls'[0]['arguments'] for Ollama.

#         arguments = data # Default for instructor output
#         if "tool_calls" in data and isinstance(data["tool_calls"], list) and data["tool_calls"]: # Ollama-like
#             tool_call = data["tool_calls"][0]
#             arguments_raw = tool_call.get("arguments", {})
#             if isinstance(arguments_raw, str):
#                 try:
#                     arguments = json.loads(arguments_raw)
#                 except json.JSONDecodeError:
#                     print(f"Error: 'arguments' field for Ollama is a string but not valid JSON: {arguments_raw[:100]}...")
#                     return 0.0
#             else:
#                 arguments = arguments_raw # Already a dict
#         elif "arguments" in data and isinstance(data["arguments"], dict): # Simpler structure with arguments dict
#              arguments = data.get("arguments", {})
#         elif not isinstance(data, dict): # If 'data' itself is not a dict (e.g. raw string from model)
#             print("Error: Response for technical metrics is not a parsable dictionary or known structure.")
#             return 0.0
#         # else: arguments = data (already set)

#         json_syntax_score = 5
#         required_fields = ["thoughts", "status", "current_info"]
#         for field in required_fields:
#             if field not in arguments:
#                 json_syntax_score -= 1
#         if "missing_info" not in arguments: # missing_info is optional in some contexts
#             json_syntax_score -= 1

#         completeness_score = 0
#         current_info = arguments.get("current_info", {})
#         # Specific checks for 'constant_data_agent' and input "Mam na imię Maria, mam 175 cm."
#         if "Maria" in user_input and current_info.get("name") == "Maria":
#             completeness_score += 1
#         if "175 cm" in user_input and current_info.get("height") == 175:
#             completeness_score += 1
        
#         missing_info = arguments.get("missing_info", [])
#         if isinstance(missing_info, list):
#             # Check if specific fields are requested if they are indeed missing
#             # This logic needs to be robust based on actual prompt instructions
#             # Example: if prompt asks for date_of_birth and it's not in current_info, it should be in missing_info
#             if any(info.get("field") == "date_of_birth" for info in missing_info):
#                  completeness_score += 0.5 # Partial credit for asking
#             if any(info.get("field") == "blood_type" for info in missing_info):
#                  completeness_score += 0.5 # Partial credit for asking
        
#         logic_score = 5
#         # This logic is also very specific
#         if arguments.get("status") != "incomplete" and "Pomiń" not in user_input and not (current_info.get("name") and current_info.get("height")):
#             logic_score -= 2 # Should be incomplete if not all primary fields filled and not skipped
#         if "Pomiń" in user_input and arguments.get("status") != "skipped":
#             logic_score -= 2
#         if not missing_info and "Pomiń" not in user_input and arguments.get("status") != "complete" and (current_info.get("name") and current_info.get("height")):
#             logic_score -=1 # If all primary data present, no missing_info, not skipped, should be complete

#         # Max score for constant_data_agent: json_syntax (5) + completeness (name, height + asking for dob, bt = 2+1=3) + logic (5) = 13
#         max_possible_score = 5 + 2 + 1 + 5 # json_syntax + name_found + height_found + (dob_asked OR bt_asked) + logic
#         # A more robust way for completeness: check if all *expected* fields are present or correctly asked for.
#         # For simplicity, using the notebook's structure for now.
#         current_total_score = json_syntax_score + completeness_score + logic_score
#         accuracy = (current_total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
#         return accuracy
#     except json.JSONDecodeError as e:
#         print(f"Error: Response is not valid JSON for technical metrics. Content: {response_str[:200]}... Error: {e}")
#         return 0.0
#     except Exception as e:
#         print(f"Error evaluating technical metrics: {e}, Response: {response_str[:200]}...")
#         return 0.0

# # --- Model Testing Functions ---
# def test_ollama_model(model_name, agent_name, prompt_text, user_test_input, reference_data_text):
#     print(f"Testing Ollama model: {model_name} for agent: {agent_name}")
#     with mlflow.start_run(run_name=f"ollama_{model_name}_{agent_name}"):
#         mlflow.log_param("model_type", "ollama")
#         mlflow.log_param("model_name", model_name)
#         mlflow.log_param("agent_name", agent_name)
#         mlflow.log_text(prompt_text, artifact_file=f"prompt_{agent_name}.txt")
#         mlflow.log_text(user_test_input, artifact_file=f"input_{agent_name}.txt")
#         if reference_data_text:
#             mlflow.log_text(reference_data_text, artifact_file=f"reference_{agent_name}.txt")

#         try:
#             print(f"Pulling Ollama model: {model_name}...")
#             # Stream output for pull to see progress, but don't fail script if pull has non-error output to stderr
#             pull_process = subprocess.Popen(["ollama", "pull", model_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#             for line in iter(pull_process.stdout.readline, ''):
#                 print(line, end='')
#             pull_process.wait()
#             if pull_process.returncode != 0:
#                  print(f"Error pulling Ollama model {model_name}. Return code: {pull_process.returncode}")
#                  # mlflow.log_metric("error_pulling", 1)
#                  # return None # Decide if to continue if pull fails (model might exist)
#         except Exception as e:
#             print(f"Exception during Ollama model pull {model_name}: {e}")
#             # mlflow.log_metric("error_pulling_exception", 1)
#             # return None

#         payload = {
#             "model": model_name,
#             "prompt": f"{prompt_text}\n\nUser input: {user_test_input}",
#             "stream": False,
#             "format": "json" 
#         }
        
#         start_time = time.time()
#         try:
#             # Using ollama run with --format json (if available in CLI) or /api/generate
#             # The notebook used 'ollama generate' which implies API usage if format:json is a feature of that endpoint directly
#             # For CLI, it might be 'ollama run modelname "prompt" --format json'
#             # Sticking to 'ollama generate' as per notebook, assuming it's a wrapper for the API call
#             process = subprocess.run(
#                 ["ollama", "generate"], 
#                 input=json.dumps(payload), 
#                 capture_output=True, 
#                 text=True, 
#                 check=True,
#                 timeout=self.config.get('model', {}).get('etag_timeout', 300) # Added timeout
#             )
#             response_json_str = process.stdout
#             try:
#                 # Ollama's format:json often wraps the actual JSON string inside a 'response' field of a larger JSON object.
#                 full_response_obj = json.loads(response_json_str)
#                 ollama_response_content = full_response_obj.get("response")
#                 if ollama_response_content is None: # If 'response' key is not found or is null
#                     print(f"Warning: 'response' key not found or null in Ollama output for {model_name}. Using raw output: {response_json_str[:200]}...")
#                     ollama_response_content = response_json_str # Fallback
#                 elif not isinstance(ollama_response_content, str):
#                     print(f"Warning: 'response' key in Ollama output is not a string for {model_name}. Converting to JSON string. Type: {type(ollama_response_content)}")
#                     ollama_response_content = json.dumps(ollama_response_content) # Ensure it's a string for metrics

#             except json.JSONDecodeError:
#                 print(f"Warning: Ollama response for {model_name} was not a single JSON object. Raw: {response_json_str[:200]}...")
#                 ollama_response_content = response_json_str # Fallback to raw if not JSON

#         except subprocess.CalledProcessError as e:
#             print(f"Error running Ollama model {model_name}: {e.stderr}")
#             mlflow.log_metric("error_running", 1)
#             mlflow.log_text(e.stderr, f"ollama_error_{agent_name}.txt")
#             return None
#         except subprocess.TimeoutExpired:
#             print(f"Timeout running Ollama model {model_name} for agent {agent_name}.")
#             mlflow.log_metric("error_timeout", 1)
#             return None
#         except Exception as e:
#             print(f"An unexpected error occurred with Ollama model {model_name}: {e}")
#             mlflow.log_metric("error_unexpected", 1)
#             return None
        
#         end_time = time.time()
#         latency = end_time - start_time
#         mlflow.log_metric("latency_seconds", latency)

#         mlflow.log_text(ollama_response_content, f"response_raw_ollama_{agent_name}.json")

#         if reference_data_text:
#             mlflow.log_metric("bleu_score", calculate_bleu(reference_data_text, ollama_response_content))
#             mlflow.log_metric("rouge_score", calculate_rouge(reference_data_text, ollama_response_content))
#             mlflow.log_metric("semantic_similarity", calculate_semantic_similarity(reference_data_text, ollama_response_content))
        
#         tech_accuracy = evaluate_technical_metrics(ollama_response_content, user_test_input, agent_name)
#         if not pd.isna(tech_accuracy):
#             mlflow.log_metric("technical_accuracy_percent", tech_accuracy)
        
#         print(f"Finished testing Ollama model: {model_name} for agent: {agent_name}. Latency: {latency:.2f}s, Tech Acc: {tech_accuracy:.2f}%")
#         return {"model": model_name, "agent": agent_name, "response": ollama_response_content, "latency": latency, "technical_accuracy": tech_accuracy}

# def test_chatgpt_model(agent_name, prompt_text, user_test_input, reference_data_text):
#     if not patched_openai_client:
#         print("Skipping ChatGPT test as client is not initialized.")
#         return None

#     print(f"Testing ChatGPT (gpt-4) for agent: {agent_name}")
#     with mlflow.start_run(run_name=f"chatgpt_gpt-4_{agent_name}"):
#         mlflow.log_param("model_type", "chatgpt")
#         mlflow.log_param("model_name", "gpt-4") 
#         mlflow.log_param("agent_name", agent_name)
#         mlflow.log_text(prompt_text, artifact_file=f"prompt_{agent_name}.txt")
#         mlflow.log_text(user_test_input, artifact_file=f"input_{agent_name}.txt")
#         if reference_data_text:
#             mlflow.log_text(reference_data_text, artifact_file=f"reference_{agent_name}.txt")

#         response_model_class = COT_FUNCTION_CALL_PYDANTIC_MODELS.get(agent_name)
#         if not response_model_class:
#             print(f"Warning: No Pydantic model defined for agent {agent_name} with instructor. Will fetch raw response.")

#         start_time = time.time()
#         try:
#             completion = patched_openai_client.chat.completions.create(
#                 model="gpt-4", 
#                 messages=[
#                     {"role": "system", "content": prompt_text},
#                     {"role": "user", "content": user_test_input}
#                 ],
#                 response_model=response_model_class, 
#                 max_tokens=1024, 
#                 temperature=0.1 
#             )
#             if response_model_class:
#                 chatgpt_response_content = completion.model_dump_json(indent=2) 
#             else:
#                 chatgpt_response_content = completion.choices[0].message.content

#         except Exception as e:
#             print(f"Error calling OpenAI API for agent {agent_name}: {e}")
#             mlflow.log_metric("error_openai_api", 1)
#             return None
#         end_time = time.time()
#         latency = end_time - start_time
#         mlflow.log_metric("latency_seconds", latency)

#         mlflow.log_text(chatgpt_response_content, f"response_chatgpt_{agent_name}.json" if response_model_class else f"response_chatgpt_{agent_name}.txt")

#         if reference_data_text:
#             mlflow.log_metric("bleu_score", calculate_bleu(reference_data_text, chatgpt_response_content))
#             mlflow.log_metric("rouge_score", calculate_rouge(reference_data_text, chatgpt_response_content))
#             mlflow.log_metric("semantic_similarity", calculate_semantic_similarity(reference_data_text, chatgpt_response_content))

#         tech_accuracy = evaluate_technical_metrics(chatgpt_response_content, user_test_input, agent_name)
#         if not pd.isna(tech_accuracy):
#              mlflow.log_metric("technical_accuracy_percent", tech_accuracy)

#         print(f"Finished testing ChatGPT for agent: {agent_name}. Latency: {latency:.2f}s, Tech Acc: {tech_accuracy:.2f}%")
#         return {"model": "gpt-4", "agent": agent_name, "response": chatgpt_response_content, "latency": latency, "reference_data": reference_data_text, "technical_accuracy": tech_accuracy}

# # --- Main Orchestration Function ---
# def run_evaluation(model_to_test=None, test_all_local=False, test_chatgpt_flag=False):
#     results = []
#     reference_data_dict = {} 

#     if test_chatgpt_flag or test_all_local or (model_to_test and model_to_test.lower() != "chatgpt"):
#         print("\n--- Generating Reference Data from ChatGPT (gpt-4) ---")
#         for agent_name in AGENTS:
#             if agent_name not in TEST_INPUTS:
#                 print(f"Skipping reference generation for agent {agent_name}: No test input defined.")
#                 continue
#             prompt_text = load_prompt(agent_name, PROMPTS_DIR)
#             if not prompt_text:
#                 continue
            
#             user_test_input = TEST_INPUTS[agent_name]
#             print(f"Generating reference for agent: {agent_name} using ChatGPT...")
#             chatgpt_ref_result = test_chatgpt_model(agent_name, prompt_text, user_test_input, None) 
#             if chatgpt_ref_result:
#                 # results.append(chatgpt_ref_result) # Decide if reference run is part of main results
#                 reference_data_dict[agent_name] = chatgpt_ref_result["response"]
#             else:
#                 reference_data_dict[agent_name] = "" 
    
#     models_to_run_list = []
#     if model_to_test:
#         if model_to_test.lower() == "chatgpt":
#             if not test_chatgpt_flag: # If only chatgpt is specified, but flag was false
#                 print("\n--- Testing ChatGPT (gpt-4) as specified (already run if used for reference) ---")
#                 # Logic to ensure it runs if it wasn't for reference, or if user explicitly wants only chatgpt
#                 if not reference_data_dict: # If reference dict is empty, means it wasn't run
#                      for agent_name in AGENTS:
#                         if agent_name not in TEST_INPUTS: continue
#                         prompt_text = load_prompt(agent_name, PROMPTS_DIR)
#                         if not prompt_text: continue
#                         user_test_input = TEST_INPUTS[agent_name]
#                         res = test_chatgpt_model(agent_name, prompt_text, user_test_input, None)
#                         if res: results.append(res)
#         else:
#             models_to_run_list = [model_to_test] 
#     elif test_all_local:
#         models_to_run_list = LOCAL_MODELS
    
#     if models_to_run_list:
#         print(f"\n--- Testing Local Ollama Models: {', '.join(models_to_run_list)} ---")
#         for model_name in models_to_run_list:
#             print(f"\n--- Testing Local Model: {model_name} ---")
#             for agent_name in AGENTS:
#                 if agent_name not in TEST_INPUTS:
#                     print(f"Skipping agent {agent_name} for model {model_name}: No test input defined.")
#                     continue
#                 prompt_text = load_prompt(agent_name, PROMPTS_DIR)
#                 if not prompt_text:
#                     continue

#                 user_test_input = TEST_INPUTS[agent_name]
#                 reference_text = reference_data_dict.get(agent_name, "")
                
#                 ollama_result = test_ollama_model(model_name, agent_name, prompt_text, user_test_input, reference_text)
#                 if ollama_result:
#                     results.append(ollama_result)
            
#     if results:
#         df_results = pd.DataFrame(results)
#         print("\n--- Evaluation Results ---")
#         print(df_results)
#         results_filename = "model_evaluation_results.csv"
#         df_results.to_csv(results_filename, index=False)
#         print(f"\nResults saved to {results_filename}")
#         if os.path.exists(results_filename):
#              mlflow.log_artifact(results_filename)
#     else:
#         print("No evaluation runs were completed successfully to generate results.")

# # --- Script Entry Point ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run model evaluations using MLflow.")
#     parser.add_argument("--model", eval_type=str, help="Specify a single model to test (e.g., 'llama3-groq-tool-use' or 'chatgpt').")
#     parser.add_argument("--all_local", action="store_true", help="Test all defined local Ollama models.")
#     parser.add_argument("--chatgpt", action="store_true", help="Include ChatGPT (gpt-4) in the test run (also used for reference data if testing local models)." )

#     args = parser.parse_args()

#     if not args.model and not args.all_local and not args.chatgpt:
#         parser.print_help()
#         print("\nPlease specify at least one test option: --model <name>, --all_local, or --chatgpt")
#     else:
#         run_evaluation(model_to_test=args.model, test_all_local=args.all_local, test_chatgpt_flag=args.chatgpt)

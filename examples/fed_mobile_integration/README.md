# Fed-Mobile Integration with Edge LLM Lab

This directory shows how to integrate the Edge LLM Lab framework with the fed-mobile medical evaluation project.

## 🏗️ Integration Steps

### 1. Install Edge LLM Lab

```bash
# In your fed-mobile project directory
pip install edge-llm-lab

# Or install from source for development
pip install -e /path/to/edge-llm-lab
```

### 2. Create Configuration File

Create `config/medical_evaluation.yaml` in your fed-mobile project:

```yaml
source_path: "thesis_generators/source"
inference_params_path: "fed_mobile_chat_flutter/assets/config/inference_params.yaml"
evaluator_model: "gpt-4o-mini"
ollama_host: "http://localhost:11434"

agents:
  constant_data_en:
    name: "Constant Data Collection Agent"
    description: "Collects constant medical data from patients"
    prompt_path: "fed_mobile_chat_flutter/assets/prompts/multi_turn/constant_data_en.txt"
    schema_path: "fed_mobile_chat_flutter/assets/schemas/constantdataanalysiscot_schema.json"
    validation_prompt_path: "fed_mobile_chat_flutter/assets/prompts/multi_turn/validation_en.txt"
    validation_schema_path: "fed_mobile_chat_flutter/assets/schemas/constantdata_en_schema.json"
    tool_name: "send_medical_data_en"
    
  fluctuating_data:
    name: "Fluctuating Data Collection Agent"
    description: "Collects fluctuating medical data"
    prompt_path: "fed_mobile_chat_flutter/assets/prompts/multi_turn/fluctuating_data.txt"
    schema_path: "fed_mobile_chat_flutter/assets/schemas/fluctuatingdata_schema.json"
    tool_name: "send_fluctuating_data"
    
  symptom:
    name: "Symptom Collection Agent"
    description: "Collects symptom data from patients"
    prompt_path: "fed_mobile_chat_flutter/assets/prompts/multi_turn/symptoms.txt"
    schema_path: "fed_mobile_chat_flutter/assets/schemas/symptom_schema.json"
    tool_name: "send_symptom_data"
```

### 3. Update Your Evaluation Code

Replace your existing evaluation code with edge-llm-lab:

```python
# Old way (thesis_generators/base_eval.py)
from thesis_generators.base_eval import BaseEvaluation

# New way (using edge-llm-lab)
from edge_llm_lab.core.base_evaluation import BaseEvaluation
from edge_llm_lab.core.agent_config import ConfigLoader
from edge_llm_lab.evaluation.referenced_evaluator import ReferencedEvaluator

# Load your medical configuration
config = ConfigLoader.load_from_yaml("config/medical_evaluation.yaml")

# Initialize evaluator
evaluator = ReferencedEvaluator(
    model_name="granite3.1-dense:2b",
    agent_type="constant_data_en",
    config=config
)

# Your existing evaluation logic works the same way
metrics = evaluator.evaluate_with_reference(
    model_response=model_output,
    reference_response=expected_output,
    user_input=patient_input
)
```

### 4. Migration Guide

#### Replace Imports

```python
# Before
from thesis_generators.base_eval import BaseEvaluation, Agent
from thesis_generators.referenced_clean import EvalModelsReferenced

# After  
from edge_llm_lab.core.base_evaluation import BaseEvaluation
from edge_llm_lab.evaluation.referenced_evaluator import ReferencedEvaluator
```

#### Update Initialization

```python
# Before
evaluator = EvalModelsReferenced(
    model_name="granite3.1-dense:2b",
    agent=Agent.CONSTANT_DATA_EN
)

# After
config = ConfigLoader.load_from_yaml("config/medical_evaluation.yaml")
evaluator = ReferencedEvaluator(
    model_name="granite3.1-dense:2b",
    agent_type="constant_data_en",
    config=config
)
```

#### Update Method Calls

```python
# Before
metrics = evaluator.evaluate_response(response, user_input, ground_truth)

# After
metrics = evaluator.evaluate_with_reference(
    model_response=response,
    reference_response=ground_truth,
    user_input=user_input
)
```

## 🚀 Benefits of Migration

1. **Modular Design**: Framework is now reusable across projects
2. **Better Configuration**: YAML-based configuration instead of hardcoded paths
3. **Device Monitoring**: Built-in device information logging
4. **Community Contributions**: Others can contribute evaluation results
5. **Maintainability**: Cleaner, more maintainable codebase

## 📁 File Structure After Migration

```
fed-mobile/
├── config/
│   └── medical_evaluation.yaml          # Your medical configuration
├── thesis_generators/
│   ├── source/                          # Keep your existing data
│   └── evaluation_scripts.py            # Updated to use edge-llm-lab
├── fed_mobile_chat_flutter/
│   └── assets/                          # Keep your existing assets
└── requirements.txt                     # Add edge-llm-lab dependency
```

## 🔧 Custom Medical Evaluators

You can extend the framework for medical-specific evaluations:

```python
from edge_llm_lab.evaluation.referenced_evaluator import ReferencedEvaluator

class MedicalEvaluator(ReferencedEvaluator):
    def __init__(self, model_name: str, agent_type: str, config):
        super().__init__(model_name, agent_type, config)
    
    def evaluate_medical_accuracy(self, response, medical_knowledge_base):
        """Custom medical accuracy evaluation."""
        # Your medical-specific evaluation logic
        pass
    
    def check_clinical_safety(self, response):
        """Check for clinical safety issues."""
        # Safety validation logic
        pass
```

## 📊 Contributing Results

Share your medical evaluation results with the community:

1. Run evaluations with device information logging
2. Submit results to edge-llm-lab repository
3. Help others understand model performance on different devices

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure edge-llm-lab is installed
2. **Configuration Paths**: Verify paths in your config file
3. **Missing Dependencies**: Install required packages (openai, ollama, etc.)

### Getting Help

- Check edge-llm-lab documentation
- Open issues on the edge-llm-lab repository
- Review example configurations

---

**This integration maintains all your existing functionality while providing a more modular and maintainable framework.**

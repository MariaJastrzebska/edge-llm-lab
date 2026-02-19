"""Unit tests for agent configuration module."""

import pytest
import tempfile
import os
import yaml
from pathlib import Path

from edge_llm_lab.core.future_agent_config import AgentConfig, AgentType


class TestAgentType:
    """Test AgentType model."""
    
    def test_create_agent_type(self):
        """Test creating an AgentType instance."""
        agent = AgentType(
            key="test_agent",
            name="Test Agent",
            description="Test agent description",
            cot_prompt_path="path/to/cot.txt",
            validation_cot_prompt_path="path/to/validation.txt",
            display_prompt="path/to/display.txt",
            validation_schema_path="path/to/schema.json",
            cot_schema_path="path/to/cot_schema.json",
            tools_config={"temperature": 0.7}
        )
        
        assert agent.key == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.description == "Test agent description"
        assert agent.tools_config["temperature"] == 0.7
    
    def test_agent_type_with_patient_simulation(self):
        """Test AgentType with patient simulation prompt."""
        agent = AgentType(
            key="medical_agent",
            name="Medical Agent",
            description="Medical agent",
            cot_prompt_path="path/to/cot.txt",
            validation_cot_prompt_path="path/to/validation.txt",
            display_prompt="path/to/display.txt",
            validation_schema_path="path/to/schema.json",
            cot_schema_path="path/to/cot_schema.json",
            patient_simulation_prompt_path="path/to/patient.txt"
        )
        
        assert agent.patient_simulation_prompt_path == "path/to/patient.txt"
    
    def test_agent_type_default_tools_config(self):
        """Test AgentType with default empty tools_config."""
        agent = AgentType(
            key="test_agent",
            name="Test Agent",
            description="Test description",
            cot_prompt_path="path/to/cot.txt",
            validation_cot_prompt_path="path/to/validation.txt",
            display_prompt="path/to/display.txt",
            validation_schema_path="path/to/schema.json",
            cot_schema_path="path/to/cot_schema.json"
        )
        
        assert agent.tools_config == {}


class TestAgentConfig:
    """Test AgentConfig model."""
    
    def test_create_agent_config(self):
        """Test creating an AgentConfig instance."""
        agent1 = AgentType(
            key="agent1",
            name="Agent 1",
            description="First agent",
            cot_prompt_path="path1/cot.txt",
            validation_cot_prompt_path="path1/validation.txt",
            display_prompt="path1/display.txt",
            validation_schema_path="path1/schema.json",
            cot_schema_path="path1/cot_schema.json"
        )
        
        agent2 = AgentType(
            key="agent2",
            name="Agent 2",
            description="Second agent",
            cot_prompt_path="path2/cot.txt",
            validation_cot_prompt_path="path2/validation.txt",
            display_prompt="path2/display.txt",
            validation_schema_path="path2/schema.json",
            cot_schema_path="path2/cot_schema.json"
        )
        
        config = AgentConfig(agents=[agent1, agent2])
        
        assert len(config.agents) == 2
        assert config.agents[0].key == "agent1"
        assert config.agents[1].key == "agent2"
    
    def test_get_agent_by_key_found(self):
        """Test getting an agent by key when it exists."""
        agent = AgentType(
            key="test_agent",
            name="Test Agent",
            description="Test",
            cot_prompt_path="path/cot.txt",
            validation_cot_prompt_path="path/validation.txt",
            display_prompt="path/display.txt",
            validation_schema_path="path/schema.json",
            cot_schema_path="path/cot_schema.json"
        )
        
        config = AgentConfig(agents=[agent])
        found = config.get_agent_by_key("test_agent")
        
        assert found is not None
        assert found.key == "test_agent"
        assert found.name == "Test Agent"
    
    def test_get_agent_by_key_not_found(self):
        """Test getting an agent by key when it doesn't exist."""
        agent = AgentType(
            key="test_agent",
            name="Test Agent",
            description="Test",
            cot_prompt_path="path/cot.txt",
            validation_cot_prompt_path="path/validation.txt",
            display_prompt="path/display.txt",
            validation_schema_path="path/schema.json",
            cot_schema_path="path/cot_schema.json"
        )
        
        config = AgentConfig(agents=[agent])
        found = config.get_agent_by_key("nonexistent")
        
        assert found is None
    
    def test_load_from_yaml(self):
        """Test loading AgentConfig from YAML file."""
        # Create temporary YAML file
        yaml_content = {
            "agents": [
                {
                    "key": "test_agent",
                    "name": "Test Agent",
                    "description": "Test agent description",
                    "cot_prompt_path": "prompts/cot.txt",
                    "validation_cot_prompt_path": "prompts/validation.txt",
                    "display_prompt": "prompts/display.txt",
                    "validation_schema_path": "schemas/validation.json",
                    "cot_schema_path": "schemas/cot.json",
                    "tools_config": {
                        "context_size": 4096,
                        "max_tokens": 1024,
                        "temperature": 0.7
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            config = AgentConfig.load_from_yaml(temp_path)
            
            assert len(config.agents) == 1
            assert config.agents[0].key == "test_agent"
            assert config.agents[0].name == "Test Agent"
            assert config.agents[0].tools_config["context_size"] == 4096
            assert config.agents[0].tools_config["temperature"] == 0.7
        finally:
            os.unlink(temp_path)
    
    def test_load_from_yaml_multiple_agents(self):
        """Test loading multiple agents from YAML."""
        yaml_content = {
            "agents": [
                {
                    "key": "agent1",
                    "name": "Agent 1",
                    "description": "First agent",
                    "cot_prompt_path": "path1/cot.txt",
                    "validation_cot_prompt_path": "path1/validation.txt",
                    "display_prompt": "path1/display.txt",
                    "validation_schema_path": "path1/schema.json",
                    "cot_schema_path": "path1/cot_schema.json"
                },
                {
                    "key": "agent2",
                    "name": "Agent 2",
                    "description": "Second agent",
                    "cot_prompt_path": "path2/cot.txt",
                    "validation_cot_prompt_path": "path2/validation.txt",
                    "display_prompt": "path2/display.txt",
                    "validation_schema_path": "path2/schema.json",
                    "cot_schema_path": "path2/cot_schema.json",
                    "patient_simulation_prompt_path": "path2/patient.txt"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            config = AgentConfig.load_from_yaml(temp_path)
            
            assert len(config.agents) == 2
            
            agent1 = config.get_agent_by_key("agent1")
            assert agent1 is not None
            assert agent1.patient_simulation_prompt_path is None
            
            agent2 = config.get_agent_by_key("agent2")
            assert agent2 is not None
            assert agent2.patient_simulation_prompt_path == "path2/patient.txt"
        finally:
            os.unlink(temp_path)
    
    def test_load_from_yaml_file_not_found(self):
        """Test loading from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            AgentConfig.load_from_yaml("nonexistent.yaml")
    
    def test_empty_agents_list(self):
        """Test creating AgentConfig with empty agents list."""
        config = AgentConfig(agents=[])
        
        assert len(config.agents) == 0
        assert config.get_agent_by_key("any_key") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""Integration tests for configuration loading."""

import pytest
import tempfile
import os
from pathlib import Path

from edge_llm_lab.core.future_agent_config import AgentConfig


class TestConfigurationIntegration:
    """Test end-to-end configuration loading."""
    
    def test_load_example_desktop_config(self):
        """Test loading the example desktop configuration."""
        # Get path to example config
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "examples" / "desktop" / "config" / "evaluation_config.yaml"
        
        if not config_path.exists():
            pytest.skip(f"Example config not found: {config_path}")
        
        # Load configuration
        config = AgentConfig.load_from_yaml(str(config_path))
        
        # Verify configuration loaded correctly
        assert config is not None
        assert len(config.agents) > 0
        
        # Get first agent
        agent = config.agents[0]
        assert agent.key is not None
        assert agent.name is not None
        assert agent.cot_prompt_path is not None
        assert agent.validation_schema_path is not None
    
    def test_load_and_retrieve_agent(self):
        """Test loading config and retrieving specific agent."""
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "examples" / "desktop" / "config" / "evaluation_config.yaml"
        
        if not config_path.exists():
            pytest.skip(f"Example config not found: {config_path}")
        
        config = AgentConfig.load_from_yaml(str(config_path))
        
        # Try to get agent by key (should exist)
        first_agent_key = config.agents[0].key
        agent = config.get_agent_by_key(first_agent_key)
        
        assert agent is not None
        assert agent.key == first_agent_key
    
    def test_config_with_all_required_fields(self):
        """Test that loaded config has all required fields."""
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "examples" / "desktop" / "config" / "evaluation_config.yaml"
        
        if not config_path.exists():
            pytest.skip(f"Example config not found: {config_path}")
        
        config = AgentConfig.load_from_yaml(str(config_path))
        
        for agent in config.agents:
            # Check required fields
            assert agent.key, "Agent must have a key"
            assert agent.name, "Agent must have a name"
            assert agent.description, "Agent must have a description"
            assert agent.cot_prompt_path, "Agent must have cot_prompt_path"
            assert agent.validation_cot_prompt_path, "Agent must have validation_cot_prompt_path"
            assert agent.display_prompt, "Agent must have display_prompt"
            assert agent.validation_schema_path, "Agent must have validation_schema_path"
            assert agent.cot_schema_path, "Agent must have cot_schema_path"
            
            # tools_config is optional but should be dict if present
            assert isinstance(agent.tools_config, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


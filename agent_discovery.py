# agent_discovery.py
import asyncio
import aiohttp
from typing import List, Dict, Any
from fastmcp import FastMCP
from pydantic import BaseModel

class ACPAgent(BaseModel):
    name: str
    description: str
    metadata: Dict[str, Any] = {}

class AgentDiscoveryTool:
    def __init__(self, acp_base_url: str = "http://localhost:8000"):
        self.acp_base_url = acp_base_url
        self.discovered_agents: Dict[str, ACPAgent] = {}
    
    async def discover_agents(self) -> List[ACPAgent]:
        """Discover all available ACP agents"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.acp_base_url}/agents") as response:
                    if response.status == 200:
                        data = await response.json()
                        agents = [ACPAgent(**agent) for agent in data.get("agents", [])]
                        
                        # Update discovered agents cache
                        for agent in agents:
                            self.discovered_agents[agent.name] = agent
                        
                        return agents
                    else:
                        print(f"Failed to discover agents: {response.status}")
                        return []
            except Exception as e:
                print(f"Error discovering agents: {e}")
                return []
    
    def get_mcp_resource_uri(self, agent_name: str) -> str:
        """Generate MCP resource URI for an ACP agent"""
        return f"acp://agents/{agent_name}"
    
    async def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed capabilities of a specific agent"""
        # This could be extended to call a capabilities endpoint
        # For now, return basic info from discovery
        agent = self.discovered_agents.get(agent_name)
        if agent:
            return {
                "name": agent.name,
                "description": agent.description,
                "metadata": agent.metadata,
                "supports_streaming": True,  # ACP supports streaming
                "supports_multimodal": True,  # ACP supports multi-modal
                "interaction_modes": ["sync", "async", "stream"]
            }
        return {}

# Integration with FastMCP
def register_discovery_tools(mcp: FastMCP, discovery: AgentDiscoveryTool):
    
    @mcp.tool()
    async def discover_acp_agents() -> str:
        """Discover all available ACP agents and register them as resources"""
        agents = await discovery.discover_agents()
        
        result = {
            "discovered_count": len(agents),
            "agents": []
        }
        
        for agent in agents:
            agent_info = {
                "name": agent.name,
                "description": agent.description,
                "resource_uri": discovery.get_mcp_resource_uri(agent.name),
                "capabilities": await discovery.get_agent_capabilities(agent.name)
            }
            result["agents"].append(agent_info)
        
        return str(result)
    
    @mcp.tool()
    async def get_agent_info(agent_name: str) -> str:
        """Get detailed information about a specific ACP agent"""
        capabilities = await discovery.get_agent_capabilities(agent_name)
        if capabilities:
            return str(capabilities)
        else:
            return f"Agent '{agent_name}' not found"

# acp_mcp_server.py
import asyncio
from fastmcp import FastMCP

# Import all components
from agent_discovery import AgentDiscoveryTool, register_discovery_tools
from message_bridge import MessageBridge, register_bridge_tools
from run_orchestrator import RunOrchestrator, register_orchestrator_tools
from agent_router import AgentRouter, register_router_tools
from interactive_manager import InteractiveManager, register_interactive_tools

class ACPMCPServer:
    def __init__(self, acp_base_url: str = "http://localhost:8000"):
        # Initialize FastMCP server
        self.mcp = FastMCP("ACP-MCP Bridge Server")
        
        # Initialize all components
        self.discovery = AgentDiscoveryTool(acp_base_url)
        self.message_bridge = MessageBridge()
        self.orchestrator = RunOrchestrator(acp_base_url)
        self.router = AgentRouter(self.discovery, self.orchestrator)
        self.interactive_manager = InteractiveManager(self.orchestrator)
        
        # Register all tools
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all component tools with FastMCP"""
        
        # Core server info tool
        @self.mcp.tool()
        async def get_server_info() -> str:
            """Get information about the ACP-MCP bridge server"""
            info = {
                "name": "ACP-MCP Bridge Server",
                "description": "Bridge between Agent Communication Protocol and Model Context Protocol",
                "components": [
                    "Agent Discovery Tool",
                    "Multi-Modal Message Bridge", 
                    "Run Orchestrator",
                    "Agent Router",
                    "Interactive Manager"
                ],
                "acp_endpoint": self.discovery.acp_base_url,
                "capabilities": [
                    "Agent discovery and registration",
                    "Multi-modal message conversion",
                    "Sync/async/streaming execution",
                    "Intelligent agent routing",
                    "Interactive agent sessions"
                ]
            }
            return str(info)
        
        # Register component tools
        register_discovery_tools(self.mcp, self.discovery)
        register_bridge_tools(self.mcp, self.message_bridge)
        register_orchestrator_tools(self.mcp, self.orchestrator)
        register_router_tools(self.mcp, self.router)
        register_interactive_tools(self.mcp, self.interactive_manager)
    
    def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
        """Start the ACP-MCP bridge server"""
        print(f"Starting ACP-MCP Bridge Server")
        print(f"Connecting to ACP server at: {self.discovery.acp_base_url}")
        print("\nAvailable tools:")
        print("- Agent Discovery & Registration")
        print("- Multi-Modal Message Bridge")
        print("- Run Orchestrator (sync/async/stream)")
        print("- Intelligent Agent Router") 
        print("- Interactive Agent Manager")
        
        if transport == "stdio":
            print("\nRunning in STDIO mode")
            self.mcp.run()
        else:
            print(f"\nServer running at http://{host}:{port}")
            self.mcp.run(transport=transport, host=host, port=port)

# Example usage and deployment
if __name__ == "__main__":
    import os
    import sys
    
    # Get ACP server URL from environment or use default
    acp_url = os.environ.get("ACP_BASE_URL", "http://localhost:8000")
    
    # Get transport mode from command line or use stdio
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    
    # Create and start the server
    server = ACPMCPServer(acp_base_url=acp_url)
    
    if transport == "http":
        server.run(transport="streamable-http", port=3000)
    else:
        server.run(transport="stdio")

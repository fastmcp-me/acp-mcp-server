# acp_mcp_server_complete.py
"""
ACP-MCP Bridge Server - Complete Implementation

A bridge server that connects Agent Communication Protocol (ACP) agents
with Model Context Protocol (MCP) clients, supporting all transport modes:
- STDIO (default)
- SSE (Server-Sent Events)  
- Streamable HTTP
"""

import asyncio
import argparse
import os
import sys
from typing import Optional
from fastmcp import FastMCP

# Import all components
from .agent_discovery import AgentDiscoveryTool, register_discovery_tools
from .message_bridge import MessageBridge, register_bridge_tools
from .run_orchestrator import RunOrchestrator, register_orchestrator_tools
from .agent_router import AgentRouter, register_router_tools
from .interactive_manager import InteractiveManager, register_interactive_tools

class ACPMCPServer:
    """ACP-MCP Bridge Server supporting multiple transport modes"""
    
    def __init__(self, acp_base_url: str = "http://localhost:8000"):
        # Initialize FastMCP server with error handling
        self.mcp = FastMCP(
            name="ACP-MCP Bridge Server",
            mask_error_details=False  # Show detailed errors for debugging
        )
        
        # Initialize all components
        self.acp_base_url = acp_base_url
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
                "version": "2.1.0",
                "components": [
                    "Agent Discovery Tool",
                    "Multi-Modal Message Bridge", 
                    "Run Orchestrator",
                    "Agent Router",
                    "Interactive Manager"
                ],
                "acp_endpoint": self.discovery.acp_base_url,
                "supported_transports": [
                    "stdio (default)",
                    "sse (Server-Sent Events)",
                    "streamable-http"
                ],
                "capabilities": [
                    "Agent discovery and registration",
                    "Multi-modal message conversion",
                    "Sync/async/streaming execution",
                    "Intelligent agent routing",
                    "Interactive agent sessions",
                    "Multiple transport protocol support"
                ]
            }
            return str(info)
        
        # Register component tools
        register_discovery_tools(self.mcp, self.discovery)
        register_bridge_tools(self.mcp, self.message_bridge)
        register_orchestrator_tools(self.mcp, self.orchestrator)
        register_router_tools(self.mcp, self.router)
        register_interactive_tools(self.mcp, self.interactive_manager)
    
    def run(
        self, 
        transport: str = "stdio", 
        host: str = "127.0.0.1", 
        port: int = 8000,
        path: str = "/mcp"
    ):
        """
        Start the ACP-MCP bridge server with specified transport
        
        Args:
            transport: Transport mode ('stdio', 'sse', 'streamable-http')
            host: Host address (for HTTP transports)
            port: Port number (for HTTP transports)
            path: URL path (for HTTP transports)
        """
        print(f"🚀 Starting ACP-MCP Bridge Server v2.1.0")
        print(f"🔗 Connecting to ACP server at: {self.acp_base_url}")
        print(f"📡 Transport: {transport}")
        
        print("\n🛠️ Available tools:")
        print("- ✅ Agent Discovery & Registration")
        print("- ✅ Multi-Modal Message Bridge")
        print("- ✅ Run Orchestrator (sync/async/stream)")
        print("- ✅ Intelligent Agent Router") 
        print("- ✅ Interactive Agent Manager")
        
        print(f"\n🌐 Supported transports:")
        print("- 📱 STDIO (default, for Claude Desktop)")
        print("- 🔄 SSE (Server-Sent Events)")
        print("- 🌍 Streamable HTTP")
        
        if transport == "stdio":
            print("\n📡 Running in STDIO mode (compatible with Claude Desktop)")
            self.mcp.run()
        elif transport == "sse":
            print(f"\n🔄 Running SSE server at http://{host}:{port}{path}")
            self.mcp.run(transport="sse", host=host, port=port, path=path)
        elif transport == "streamable-http":
            print(f"\n🌍 Running HTTP server at http://{host}:{port}{path}")
            self.mcp.run(transport="streamable-http", host=host, port=port, path=path)
        else:
            raise ValueError(f"Unsupported transport: {transport}. Use 'stdio', 'sse', or 'streamable-http'")


def create_cli():
    """Create command line interface"""
    parser = argparse.ArgumentParser(
        description="ACP-MCP Bridge Server - Connect ACP agents with MCP clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with STDIO (default, for Claude Desktop)
  acp-mcp-server

  # Run with SSE transport
  acp-mcp-server --transport sse --port 8000

  # Run with HTTP transport  
  acp-mcp-server --transport streamable-http --host 0.0.0.0 --port 9000

  # Connect to different ACP server
  acp-mcp-server --acp-url http://localhost:8001
        """
    )
    
    parser.add_argument(
        "--transport", 
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for HTTP transports (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for HTTP transports (default: 8000)"
    )
    
    parser.add_argument(
        "--path",
        default="/mcp",
        help="URL path for HTTP transports (default: /mcp)"
    )
    
    parser.add_argument(
        "--acp-url",
        default=None,
        help="ACP server URL (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="ACP-MCP Bridge Server v2.1.0"
    )
    
    return parser


def main():
    """Main entry point for the CLI"""
    parser = create_cli()
    args = parser.parse_args()
    
    # Get ACP server URL from environment or command line
    acp_url = (
        args.acp_url or 
        os.environ.get("ACP_BASE_URL") or 
        "http://localhost:8000"
    )
    
    try:
        # Create and start the server
        server = ACPMCPServer(acp_base_url=acp_url)
        server.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down ACP-MCP Bridge Server")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


# Support both direct execution and importable usage
if __name__ == "__main__":
    main()

# ACP-MCP-Server

![](https://badge.mcpx.dev?type=server "MCP Server")
[![PyPI version](https://badge.fury.io/py/acp-mcp-server.svg)](https://badge.fury.io/py/acp-mcp-server)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A bridge server that connects **Agent Communication Protocol (ACP)** agents with **Model Context Protocol (MCP)** clients, enabling seamless integration between ACP-based AI agents and MCP-compatible tools like Claude Desktop.

## ✨ Features

- 🔄 **Protocol Bridge**: Seamlessly connects ACP agents with MCP clients
- 🚀 **Multiple Transports**: Supports STDIO, SSE, and Streamable HTTP
- 🤖 **Agent Discovery**: Automatic discovery and registration of ACP agents
- 🧠 **Smart Routing**: Intelligent routing of requests to appropriate agents
- 🔄 **Async Support**: Full support for synchronous and asynchronous operations
- 💬 **Interactive Sessions**: Support for multi-turn agent interactions
- 🌐 **Multi-Modal**: Handle text, images, and other content types

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install acp-mcp-server

# Or use uvx for isolated execution
uvx acp-mcp-server
```

### Basic Usage

```bash
# Run with STDIO (default, for Claude Desktop)
acp-mcp-server

# Run with SSE transport
acp-mcp-server --transport sse --port 8000

# Run with HTTP transport
acp-mcp-server --transport streamable-http --host 0.0.0.0 --port 9000

# Connect to different ACP server
acp-mcp-server --acp-url http://localhost:8001
```

### Using with Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "acp-bridge": {
      "command": "uvx",
      "args": ["acp-mcp-server"]
    }
  }
}
```

## 📋 Requirements

- Python 3.11+
- Running ACP server with agents
- FastMCP for protocol implementation

## 🔧 Configuration

### Environment Variables

- `ACP_BASE_URL`: ACP server URL (default: `http://localhost:8000`)

### Command Line Options

```
usage: acp-mcp-server [-h] [--transport {stdio,sse,streamable-http}] [--host HOST] [--port PORT] [--path PATH] [--acp-url ACP_URL] [--version]

options:
  -h, --help            show this help message and exit
  --transport {stdio,sse,streamable-http}
                        Transport protocol (default: stdio)
  --host HOST           Host address for HTTP transports (default: 127.0.0.1)
  --port PORT           Port number for HTTP transports (default: 8000)
  --path PATH           URL path for HTTP transports (default: /mcp)
  --acp-url ACP_URL     ACP server URL (default: http://localhost:8000)
  --version             show program's version number and exit
```

## 🛠️ Available Tools

The bridge server provides several MCP tools:

### Agent Management
- `discover_acp_agents`: Discover available ACP agents
- `get_agent_info`: Get detailed information about specific agents

### Agent Execution
- `run_acp_agent`: Execute agents in sync/async modes
- `get_async_run_result`: Retrieve results from async executions
- `list_active_runs`: List all active agent runs

### Smart Routing
- `smart_route_request`: Intelligently route requests to best agents
- `test_routing`: Test routing logic without execution
- `add_routing_rule`: Add custom routing rules
- `list_routing_strategies`: View all routing strategies

### Interactive Sessions
- `start_interactive_agent`: Start interactive agent sessions
- `provide_user_input`: Provide input to waiting agents
- `list_pending_interactions`: View pending interactions

### Message Processing
- `convert_acp_message`: Convert between ACP and MCP formats
- `analyze_message_content`: Analyze message structure and content

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │    │  ACP-MCP Bridge │    │   ACP Agents    │
│ (Claude Desktop)│◄──►│     Server      │◄──►│ (echo, chat,    │
│                 │    │                 │    │  translate...)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘

### Helper Script

Use the `run_server.py` helper script for convenient server management:

```bash
# Run with local installation
./run_server.py --preset claude  # For Claude Desktop (STDIO)
./run_server.py --preset http    # HTTP server on port 9000
./run_server.py --preset sse     # SSE server on port 8000
./run_server.py --preset dev     # Development mode

# Run with uvx
./run_server.py --method uvx --transport sse --port 8000

# Run with Docker
./run_server.py --method docker --transport streamable-http --port 9000

# Custom configuration
./run_server.py --transport sse --host 0.0.0.0 --port 8080 --acp-url http://my-acp:8001
```

       │                         │                       │
   MCP Protocol            Protocol Bridge         ACP Protocol
  (STDIO/SSE/HTTP)        (FastMCP + aiohttp)    (HTTP/WebSocket)
```

## 🔌 Transport Modes

### STDIO (Default)
Perfect for Claude Desktop integration:
```bash
acp-mcp-server
```

### SSE (Server-Sent Events)
For web applications and streaming:
```bash
acp-mcp-server --transport sse --port 8000
```

### Streamable HTTP
For REST API integration:
```bash
acp-mcp-server --transport streamable-http --port 9000
```

## 🧪 Development

### Setup Development Environment

```bash
git clone https://github.com/GongRzhe/ACP-MCP-Server
cd ACP-MCP-Server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black acp_mcp_server/
isort acp_mcp_server/
```

## 📚 Examples

### Basic Agent Execution

```python
from acp_mcp_server import ACPMCPServer

# Create server instance
server = ACPMCPServer("http://localhost:8000")

# Run with STDIO for Claude Desktop
server.run()

# Or run with HTTP for web integration
server.run(transport="streamable-http", port=9000)
```

### Custom ACP Integration

```python
# Connect to custom ACP server
server = ACPMCPServer("http://my-acp-server:8001")
server.run(transport="sse", port=8080)
```

## 🐳 Docker

### Quick Start with Docker

```bash
# Build the image
docker build -t acp-mcp-server .

# Run with Streamable HTTP transport
docker run -p 9000:9000 acp-mcp-server

# Run with SSE transport
docker run -p 8000:8000 acp-mcp-server \
  --transport sse --host 0.0.0.0 --port 8000

# Connect to custom ACP server
docker run -p 9000:9000 -e ACP_BASE_URL=http://my-acp-server:8001 acp-mcp-server
```

### Using Docker Compose

```bash
# Run HTTP transport service
docker-compose up acp-mcp-http

# Run SSE transport service
docker-compose up acp-mcp-sse

# Run both services
docker-compose up

# Run development mode with live code reload
docker-compose --profile dev up acp-mcp-dev
```

### Production Docker Image

For production deployments, use the multi-stage Dockerfile:

```bash
# Build production image
docker build -f Dockerfile.prod -t acp-mcp-server:prod .

# Run production container
docker run -d \
  --name acp-mcp-server \
  --restart unless-stopped \
  -p 9000:9000 \
  -e ACP_BASE_URL=http://your-acp-server:8000 \
  acp-mcp-server:prod
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [FastMCP](https://github.com/jlowin/fastmcp) - Fast, Pythonic MCP server framework
- [ACP SDK](https://github.com/i-am-bee/acp) - Agent Communication Protocol SDK
- [Claude Desktop](https://claude.ai/desktop) - AI assistant with MCP support

## 📞 Support

- 🐛 [Report Issues](https://github.com/GongRzhe/ACP-MCP-Server/issues)
- 💬 [Discussions](https://github.com/GongRzhe/ACP-MCP-Server/discussions)
- 📖 [Documentation](https://github.com/GongRzhe/ACP-MCP-Server#readme)

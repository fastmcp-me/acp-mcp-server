#!/usr/bin/env python3
"""
Run ACP-MCP Server with various configurations

This script provides a convenient way to run the ACP-MCP Server with different
transport modes and configurations.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import acp_mcp_server
        return True
    except ImportError:
        return False


def run_local(args):
    """Run the server from local source"""
    project_root = Path(__file__).parent
    server_module = project_root / "acp_mcp_server"
    
    if not server_module.exists():
        print("Error: acp_mcp_server module not found in current directory")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "acp_mcp_server"]
    
    # Add arguments
    if args.transport:
        cmd.extend(["--transport", args.transport])
    if args.host:
        cmd.extend(["--host", args.host])
    if args.port:
        cmd.extend(["--port", str(args.port)])
    if args.path:
        cmd.extend(["--path", args.path])
    if args.acp_url:
        cmd.extend(["--acp-url", args.acp_url])
    
    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)


def run_uvx(args):
    """Run the server using uvx"""
    cmd = ["uvx", "acp-mcp-server"]
    
    # Add arguments
    if args.transport:
        cmd.extend(["--transport", args.transport])
    if args.host:
        cmd.extend(["--host", args.host])
    if args.port:
        cmd.extend(["--port", str(args.port)])
    if args.path:
        cmd.extend(["--path", args.path])
    if args.acp_url:
        cmd.extend(["--acp-url", args.acp_url])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_docker(args):
    """Run the server using Docker"""
    image_name = "acp-mcp-server"
    
    # Check if image exists
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True
    )
    
    if not result.stdout.strip():
        print(f"Docker image '{image_name}' not found. Building...")
        subprocess.run(["docker", "build", "-t", image_name, "."])
    
    # Build docker run command
    cmd = [
        "docker", "run", "--rm", "-it",
        "--name", "acp-mcp-server"
    ]
    
    # Add port mapping
    port = args.port or 8000
    cmd.extend(["-p", f"{port}:{port}"])
    
    # Add environment variables
    if args.acp_url:
        cmd.extend(["-e", f"ACP_BASE_URL={args.acp_url}"])
    
    cmd.append(image_name)
    
    # Add command arguments
    if args.transport:
        cmd.extend(["--transport", args.transport])
    cmd.extend(["--host", "0.0.0.0"])  # Always use 0.0.0.0 in Docker
    if args.port:
        cmd.extend(["--port", str(args.port)])
    if args.path:
        cmd.extend(["--path", args.path])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run ACP-MCP Server with various configurations"
    )
    
    # Runtime method
    parser.add_argument(
        "--method",
        choices=["local", "uvx", "docker"],
        default="local",
        help="Method to run the server (default: local)"
    )
    
    # Server options
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport protocol"
    )
    parser.add_argument(
        "--host",
        help="Host address for HTTP transports"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number for HTTP transports"
    )
    parser.add_argument(
        "--path",
        help="URL path for HTTP transports"
    )
    parser.add_argument(
        "--acp-url",
        help="ACP server URL"
    )
    
    # Preset configurations
    parser.add_argument(
        "--preset",
        choices=["claude", "http", "sse", "dev"],
        help="Use a preset configuration"
    )
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset == "claude":
        args.transport = "stdio"
    elif args.preset == "http":
        args.transport = "streamable-http"
        args.host = args.host or "127.0.0.1"
        args.port = args.port or 9000
    elif args.preset == "sse":
        args.transport = "sse"
        args.host = args.host or "127.0.0.1"
        args.port = args.port or 8000
    elif args.preset == "dev":
        args.transport = "streamable-http"
        args.host = args.host or "127.0.0.1"
        args.port = args.port or 8080
        args.acp_url = args.acp_url or "http://localhost:8001"
    
    # Check dependencies for local method
    if args.method == "local" and not check_dependencies():
        print("Warning: acp_mcp_server not installed. Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
    
    # Run with selected method
    if args.method == "local":
        run_local(args)
    elif args.method == "uvx":
        run_uvx(args)
    elif args.method == "docker":
        run_docker(args)


if __name__ == "__main__":
    main()

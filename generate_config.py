#!/usr/bin/env python3
"""
Generate Claude Desktop configuration for ACP-MCP-Server

This script generates the configuration JSON for Claude Desktop to use the ACP-MCP-Server.
It supports both local deployment and PyPI installation via uvx.

Usage:
    python generate_config.py [--local | --pypi] [--project-path PATH]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def generate_local_config(project_path: Path) -> dict:
    """Generate configuration for local deployment"""
    python_path = project_path / ".venv" / "bin" / "python"
    server_path = project_path / "acp_mcp_server" / "server.py"
    
    # Check if paths exist
    if not python_path.exists():
        print(f"Warning: Python path not found: {python_path}")
        print("Make sure you have created a virtual environment with: python -m venv .venv")
    
    if not server_path.exists():
        print(f"Error: Server script not found: {server_path}")
        sys.exit(1)
    
    return {
        "ACP-MCP-Server": {
            "command": str(python_path),
            "args": [str(server_path)],
            "env": {
                "PYTHONPATH": str(project_path)
            }
        }
    }


def generate_pypi_config() -> dict:
    """Generate configuration for PyPI installation via uvx"""
    return {
        "ACP-MCP-Server": {
            "command": "uvx",
            "args": ["acp-mcp-server"]
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Claude Desktop configuration for ACP-MCP-Server"
    )
    
    # Deployment type
    deployment_group = parser.add_mutually_exclusive_group()
    deployment_group.add_argument(
        "--local",
        action="store_true",
        help="Generate configuration for local deployment (default)"
    )
    deployment_group.add_argument(
        "--pypi",
        action="store_true",
        help="Generate configuration for PyPI installation via uvx"
    )
    
    # Project path for local deployment
    parser.add_argument(
        "--project-path",
        type=Path,
        default=Path.cwd(),
        help="Path to ACP-MCP-Server project (default: current directory)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: print to stdout)"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )
    
    args = parser.parse_args()
    
    # Default to local if neither is specified
    if not args.local and not args.pypi:
        args.local = True
    
    # Generate configuration
    if args.pypi:
        config = generate_pypi_config()
        print("Generated configuration for PyPI installation (uvx)")
    else:
        config = generate_local_config(args.project_path.resolve())
        print(f"Generated configuration for local deployment at: {args.project_path.resolve()}")
    
    # Format JSON
    if args.pretty:
        json_output = json.dumps(config, indent=2)
    else:
        json_output = json.dumps(config)
    
    # Output
    if args.output:
        args.output.write_text(json_output + "\n")
        print(f"\nConfiguration written to: {args.output}")
    else:
        print("\nConfiguration:")
        print(json_output)
        
        # Add instructions
        print("\n" + "="*60)
        print("To use this configuration:")
        print("1. Open Claude Desktop settings")
        print("2. Go to 'Developer' tab")
        print("3. Find 'Model Context Protocol' section")
        print("4. Click 'Edit Config'")
        print("5. Add the above configuration to the 'mcpServers' object")
        print("="*60)
        
        # Platform-specific paths
        print("\nClaude Desktop configuration file locations:")
        print("- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
        print("- Windows: %APPDATA%\\Claude\\claude_desktop_config.json")
        print("- Linux: ~/.config/Claude/claude_desktop_config.json")


if __name__ == "__main__":
    main()

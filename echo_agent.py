# echo_agent.py
"""
Simple echo agent for testing the ACP-MCP bridge.
This agent demonstrates basic ACP functionality.
"""

from acp_sdk.server import Server
from acp_sdk.models import Message, MessagePart

server = Server()

@server.agent()
async def echo(input: list[Message]):
    """Simple echo agent that returns the input messages"""
    for message in input:
        # Echo each message back
        yield message

@server.agent()
async def echo_with_prefix(input: list[Message]):
    """Echo agent that adds a prefix to each message"""
    for message in input:
        # Create a new message with prefix
        new_parts = []
        for part in message.parts:
            if hasattr(part, 'content') and part.content:
                new_content = f"ECHO: {part.content}"
                new_part = MessagePart(
                    content=new_content,
                    content_type=part.content_type if hasattr(part, 'content_type') else "text/plain"
                )
                new_parts.append(new_part)
            else:
                new_parts.append(part)
        
        # Create new message with modified parts
        new_message = Message(parts=new_parts)
        yield new_message

if __name__ == "__main__":
    # Start the ACP server
    server.run(port=8000)

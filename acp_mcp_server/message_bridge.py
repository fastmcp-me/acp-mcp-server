# message_bridge.py
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel
from fastmcp import FastMCP
import aiohttp
import base64

class ACPMessagePart(BaseModel):
    name: Optional[str] = None
    content_type: str
    content: Optional[str] = None
    content_encoding: Optional[str] = "plain"
    content_url: Optional[str] = None

class ACPMessage(BaseModel):
    parts: List[ACPMessagePart]

class MCPContent(BaseModel):
    type: str  # "text", "image", "resource"
    text: str = None
    data: str = None
    mimeType: str = None

class MessageBridge:
    def __init__(self):
        self.supported_content_types = {
            "text/plain": "text",
            "text/markdown": "text", 
            "image/jpeg": "image",
            "image/png": "image",
            "image/gif": "image",
            "application/json": "text"
        }
    
    async def acp_to_mcp(self, acp_messages: List[ACPMessage]) -> List[MCPContent]:
        """Convert ACP messages to MCP content format"""
        mcp_content = []
        
        for message in acp_messages:
            for part in message.parts:
                content = await self._convert_message_part(part)
                if content:
                    mcp_content.append(content)
        
        return mcp_content
    
    async def mcp_to_acp(self, text_content: str) -> List[ACPMessage]:
        """Convert MCP text to ACP message format"""
        # Create properly formatted ACP message
        message_part = ACPMessagePart(
            name=None,
            content_type="text/plain",
            content=text_content,
            content_encoding="plain",
            content_url=None
        )
        
        return [ACPMessage(parts=[message_part])]
    
    async def _convert_message_part(self, part: ACPMessagePart) -> MCPContent:
        """Convert a single ACP message part to MCP content"""
        content_type = part.content_type
        
        if content_type.startswith("text/"):
            return MCPContent(
                type="text",
                text=part.content,
                mimeType=content_type
            )
        
        elif content_type.startswith("image/"):
            # Handle image content
            if part.content_url:
                # Download image from URL
                image_data = await self._download_content(part.content_url)
                if image_data:
                    return MCPContent(
                        type="image", 
                        data=base64.b64encode(image_data).decode(),
                        mimeType=content_type
                    )
            else:
                # Direct image data
                return MCPContent(
                    type="image",
                    data=part.content,
                    mimeType=content_type
                )
        
        elif content_type == "application/json":
            return MCPContent(
                type="text",
                text=part.content,
                mimeType=content_type
            )
        
        else:
            # Fallback to text representation
            return MCPContent(
                type="text", 
                text=str(part.content),
                mimeType="text/plain"
            )
    
    async def _download_content(self, url: str) -> bytes:
        """Download content from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
        except Exception as e:
            print(f"Error downloading content from {url}: {e}")
        return None

# Integration with FastMCP
def register_bridge_tools(mcp: FastMCP, bridge: MessageBridge):
    
    @mcp.tool()
    async def convert_acp_message(acp_message_json: str) -> str:
        """Convert ACP message format to MCP-compatible format"""
        try:
            import json
            message_data = json.loads(acp_message_json)
            acp_message = ACPMessage(**message_data)
            
            mcp_content = await bridge.acp_to_mcp([acp_message])
            
            return json.dumps([content.dict() for content in mcp_content], indent=2)
            
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def analyze_message_content(acp_message_json: str) -> str:
        """Analyze the content types and structure of an ACP message"""
        try:
            import json
            message_data = json.loads(acp_message_json)
            acp_message = ACPMessage(**message_data)
            
            analysis = {
                "total_parts": len(acp_message.parts),
                "content_types": {},
                "has_urls": False,
                "encodings": set(),
                "total_size": 0
            }
            
            for part in acp_message.parts:
                content_type = part.content_type
                analysis["content_types"][content_type] = analysis["content_types"].get(content_type, 0) + 1
                analysis["encodings"].add(part.content_encoding)
                
                if part.content_url:
                    analysis["has_urls"] = True
                
                if part.content:
                    analysis["total_size"] += len(part.content)
            
            analysis["encodings"] = list(analysis["encodings"])
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error: {e}"

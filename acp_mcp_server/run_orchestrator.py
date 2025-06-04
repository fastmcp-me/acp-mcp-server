# run_orchestrator.py
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from enum import Enum
from pydantic import BaseModel
from fastmcp import FastMCP
from .message_bridge import MessageBridge, ACPMessage

class RunMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"

class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ACPRun(BaseModel):
    run_id: str
    agent_name: str
    status: RunStatus
    output: List[Dict[str, Any]] = []
    error: Optional[str] = None
    session_id: Optional[str] = None

class RunOrchestrator:
    def __init__(self, acp_base_url: str = "http://localhost:8000"):
        self.acp_base_url = acp_base_url
        self.message_bridge = MessageBridge()
        self.active_runs: Dict[str, ACPRun] = {}
    
    async def execute_agent_sync(
        self, 
        agent_name: str, 
        input_text: str,
        session_id: Optional[str] = None
    ) -> ACPRun:
        """Execute an ACP agent synchronously"""
        
        # Convert input to ACP format
        acp_messages = await self.message_bridge.mcp_to_acp(input_text)
        
        # Properly serialize ACP messages
        input_data = []
        for msg in acp_messages:
            message_dict = {
                "parts": []
            }
            for part in msg.parts:
                part_dict = {
                    "content_type": part.content_type,
                    "content": part.content
                }
                if part.name is not None:
                    part_dict["name"] = part.name
                if part.content_encoding != "plain":
                    part_dict["content_encoding"] = part.content_encoding
                if part.content_url is not None:
                    part_dict["content_url"] = part.content_url
                    
                message_dict["parts"].append(part_dict)
            input_data.append(message_dict)
        
        payload = {
            "agent_name": agent_name,
            "input": input_data,
            "mode": "sync"
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        print(f"Sending payload to ACP server: {json.dumps(payload, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.acp_base_url}/runs",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    response_text = await response.text()
                    print(f"ACP server response ({response.status}): {response_text}")
                    
                    if response.status == 200:
                        result = json.loads(response_text)
                        run = ACPRun(**result)
                        self.active_runs[run.run_id] = run
                        return run
                    else:
                        error_msg = f"ACP request failed: {response.status} - {response_text}"
                        return ACPRun(
                            run_id="error",
                            agent_name=agent_name,
                            status=RunStatus.FAILED,
                            error=error_msg
                        )
                        
            except Exception as e:
                return ACPRun(
                    run_id="error",
                    agent_name=agent_name,
                    status=RunStatus.FAILED,
                    error=str(e)
                )
    
    async def execute_agent_async(
        self,
        agent_name: str,
        input_text: str,
        session_id: Optional[str] = None
    ) -> str:
        """Start an asynchronous ACP agent execution"""
        
        acp_messages = await self.message_bridge.mcp_to_acp(input_text)
        
        # Properly serialize ACP messages
        input_data = []
        for msg in acp_messages:
            message_dict = {
                "parts": []
            }
            for part in msg.parts:
                part_dict = {
                    "content_type": part.content_type,
                    "content": part.content
                }
                if part.name is not None:
                    part_dict["name"] = part.name
                if part.content_encoding != "plain":
                    part_dict["content_encoding"] = part.content_encoding
                if part.content_url is not None:
                    part_dict["content_url"] = part.content_url
                    
                message_dict["parts"].append(part_dict)
            input_data.append(message_dict)
        
        payload = {
            "agent_name": agent_name,
            "input": input_data,
            "mode": "async"
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.acp_base_url}/runs",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status in [200, 202]:  # Accept both 200 and 202 for async operations
                        result = await response.json()
                        run_id = result.get("run_id")
                        
                        # Store partial run info
                        run = ACPRun(
                            run_id=run_id,
                            agent_name=agent_name,
                            status=RunStatus.CREATED
                        )
                        self.active_runs[run_id] = run
                        
                        return run_id
                    else:
                        response_text = await response.text()
                        raise Exception(f"Failed to start async run: {response.status} - {response_text}")
                        
            except Exception as e:
                raise Exception(f"Error starting async run: {e}")
    
    async def get_run_status(self, run_id: str) -> ACPRun:
        """Get the status of an async run"""
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.acp_base_url}/runs/{run_id}") as response:
                    if response.status == 200:
                        result = await response.json()
                        run = ACPRun(**result)
                        self.active_runs[run_id] = run
                        return run
                    else:
                        response_text = await response.text()
                        raise Exception(f"Failed to get run status: {response.status} - {response_text}")
                        
            except Exception as e:
                # Return cached run or error
                if run_id in self.active_runs:
                    return self.active_runs[run_id]
                else:
                    return ACPRun(
                        run_id=run_id,
                        agent_name="unknown",
                        status=RunStatus.FAILED,
                        error=str(e)
                    )
    
    async def execute_agent_stream(
        self,
        agent_name: str,
        input_text: str,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Execute an ACP agent with streaming"""
        
        acp_messages = await self.message_bridge.mcp_to_acp(input_text)
        
        # Properly serialize ACP messages
        input_data = []
        for msg in acp_messages:
            message_dict = {
                "parts": []
            }
            for part in msg.parts:
                part_dict = {
                    "content_type": part.content_type,
                    "content": part.content
                }
                if part.name is not None:
                    part_dict["name"] = part.name
                if part.content_encoding != "plain":
                    part_dict["content_encoding"] = part.content_encoding
                if part.content_url is not None:
                    part_dict["content_url"] = part.content_url
                    
                message_dict["parts"].append(part_dict)
            input_data.append(message_dict)
        
        payload = {
            "agent_name": agent_name,
            "input": input_data, 
            "mode": "stream"
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.acp_base_url}/runs",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                    }
                ) as response:
                    
                    if response.status == 200:
                        async for line in response.content:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data = line_str[6:]  # Remove 'data: ' prefix
                                if data and data != '[DONE]':
                                    yield data
                    else:
                        response_text = await response.text()
                        yield f"Error: Stream failed with status {response.status} - {response_text}"
                        
            except Exception as e:
                yield f"Error: {e}"

# Integration with FastMCP
def register_orchestrator_tools(mcp: FastMCP, orchestrator: RunOrchestrator):
    
    @mcp.tool()
    async def run_acp_agent(
        agent_name: str,
        input_text: str,
        mode: str = "sync",
        session_id: str = None
    ) -> str:
        """Execute an ACP agent with specified mode"""
        
        try:
            if mode == "sync":
                run = await orchestrator.execute_agent_sync(agent_name, input_text, session_id)
                
                if run.status == RunStatus.COMPLETED:
                    # Convert output back to readable format
                    if run.output:
                        # Handle ACP output format
                        output_text = ""
                        for message in run.output:
                            if isinstance(message, dict) and "parts" in message:
                                for part in message["parts"]:
                                    if isinstance(part, dict) and "content" in part:
                                        output_text += part["content"] + "\n"
                        return output_text.strip() if output_text else "Agent completed with no text output"
                    else:
                        return "Agent completed with no output"
                else:
                    return f"Error: {run.error}"
            
            elif mode == "async":
                run_id = await orchestrator.execute_agent_async(agent_name, input_text, session_id)
                return f"Started async run with ID: {run_id}"
            
            else:
                return f"Unsupported mode: {mode}. Use 'sync' or 'async'"
                
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def get_async_run_result(run_id: str) -> str:
        """Get the result of an asynchronous run"""
        
        try:
            run = await orchestrator.get_run_status(run_id)
            
            result = {
                "run_id": run.run_id,
                "agent_name": run.agent_name,
                "status": run.status,
                "has_output": len(run.output) > 0,
                "error": run.error
            }
            
            if run.status == RunStatus.COMPLETED and run.output:
                # Convert output to readable format
                output_text = ""
                for message in run.output:
                    if isinstance(message, dict) and "parts" in message:
                        for part in message["parts"]:
                            if isinstance(part, dict) and "content" in part:
                                output_text += part["content"] + "\n"
                result["output"] = output_text.strip() if output_text else "No text content"
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def list_active_runs() -> str:
        """List all active runs"""
        
        runs_info = []
        for run_id, run in orchestrator.active_runs.items():
            runs_info.append({
                "run_id": run_id,
                "agent_name": run.agent_name,
                "status": run.status,
                "has_error": run.error is not None
            })
        
        return json.dumps(runs_info, indent=2)

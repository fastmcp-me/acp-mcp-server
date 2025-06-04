# run_orchestrator.py
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from enum import Enum
from pydantic import BaseModel
from fastmcp import FastMCP
from message_bridge import MessageBridge, ACPMessage

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
        
        payload = {
            "agent_name": agent_name,
            "input": [msg.dict() for msg in acp_messages],
            "mode": "sync"
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
                    
                    if response.status == 200:
                        result = await response.json()
                        run = ACPRun(**result)
                        self.active_runs[run.run_id] = run
                        return run
                    else:
                        error_msg = f"ACP request failed: {response.status}"
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
        
        payload = {
            "agent_name": agent_name,
            "input": [msg.dict() for msg in acp_messages],
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
                    
                    if response.status == 200:
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
                        raise Exception(f"Failed to start async run: {response.status}")
                        
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
                        raise Exception(f"Failed to get run status: {response.status}")
                        
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
        
        payload = {
            "agent_name": agent_name,
            "input": [msg.dict() for msg in acp_messages], 
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
                        yield f"Error: Stream failed with status {response.status}"
                        
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
                        mcp_content = await orchestrator.message_bridge.acp_to_mcp([
                            ACPMessage(parts=run.output)
                        ])
                        result = "\n".join([content.text for content in mcp_content if content.text])
                        return result
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
                mcp_content = await orchestrator.message_bridge.acp_to_mcp([
                    ACPMessage(parts=run.output)
                ])
                result["output"] = "\n".join([content.text for content in mcp_content if content.text])
            
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

# interactive_manager.py
import json
import asyncio
from typing import Dict, Optional, Any
from pydantic import BaseModel
from fastmcp import FastMCP
from .run_orchestrator import RunOrchestrator
from enum import Enum

class InteractionState(str, Enum):
    WAITING = "waiting"
    RESUMED = "resumed"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"

class PendingInteraction(BaseModel):
    run_id: str
    agent_name: str
    session_id: Optional[str]
    await_message: str
    timestamp: float
    timeout_seconds: int = 300  # 5 minutes default

class InteractiveManager:
    def __init__(self, orchestrator: RunOrchestrator):
        self.orchestrator = orchestrator
        self.pending_interactions: Dict[str, PendingInteraction] = {}
        self.interaction_results: Dict[str, Any] = {}
    
    async def start_interactive_agent(
        self,
        agent_name: str,
        initial_input: str,
        session_id: Optional[str] = None,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """Start an interactive agent that may require user input"""
        
        try:
            # Start the agent execution
            run = await self.orchestrator.execute_agent_sync(
                agent_name=agent_name,
                input_text=initial_input,
                session_id=session_id
            )
            
            # Check if agent is waiting for input
            if hasattr(run, 'await_request') and run.await_request:
                # Agent is waiting for input
                pending = PendingInteraction(
                    run_id=run.run_id,
                    agent_name=agent_name,
                    session_id=session_id,
                    await_message=run.await_request.get('message', 'Agent is waiting for input'),
                    timestamp=asyncio.get_event_loop().time(),
                    timeout_seconds=timeout_seconds
                )
                
                self.pending_interactions[run.run_id] = pending
                
                return {
                    "status": "awaiting_input",
                    "run_id": run.run_id,
                    "message": pending.await_message,
                    "timeout_seconds": timeout_seconds
                }
            
            else:
                # Agent completed normally
                output = ""
                if run.output:
                    # Handle ACP output format - run.output is already a list of messages
                    output_text = ""
                    for message in run.output:
                        if isinstance(message, dict) and "parts" in message:
                            for part in message["parts"]:
                                if isinstance(part, dict) and "content" in part:
                                    output_text += part["content"] + "\n"
                    output = output_text.strip() if output_text else "No text content"
                
                return {
                    "status": "completed",
                    "run_id": run.run_id,
                    "output": output,
                    "error": run.error
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def resume_interactive_agent(
        self,
        run_id: str,
        user_input: str
    ) -> Dict[str, Any]:
        """Resume an agent that was waiting for user input"""
        
        if run_id not in self.pending_interactions:
            return {
                "status": "error",
                "error": f"No pending interaction found for run_id: {run_id}"
            }
        
        pending = self.pending_interactions[run_id]
        
        # Check timeout
        current_time = asyncio.get_event_loop().time()
        if current_time - pending.timestamp > pending.timeout_seconds:
            del self.pending_interactions[run_id]
            return {
                "status": "timeout",
                "error": "Interaction timed out"
            }
        
        try:
            # Resume the agent with user input
            # Note: This is a simplified version - actual ACP resume implementation may vary
            resume_payload = {
                "run_id": run_id,
                "resume_input": user_input
            }
            
            # For now, we'll simulate resume by starting a new session
            # In a real implementation, this would use ACP's resume endpoint
            run = await self.orchestrator.execute_agent_sync(
                agent_name=pending.agent_name,
                input_text=user_input,
                session_id=pending.session_id
            )
            
            # Clean up pending interaction
            del self.pending_interactions[run_id]
            
            # Check if agent needs more input
            if hasattr(run, 'await_request') and run.await_request:
                # Agent needs more input
                new_pending = PendingInteraction(
                    run_id=run.run_id,
                    agent_name=pending.agent_name,
                    session_id=pending.session_id,
                    await_message=run.await_request.get('message', 'Agent is waiting for more input'),
                    timestamp=current_time,
                    timeout_seconds=pending.timeout_seconds
                )
                
                self.pending_interactions[run.run_id] = new_pending
                
                return {
                    "status": "awaiting_more_input",
                    "run_id": run.run_id,
                    "message": new_pending.await_message
                }
            
            else:
                # Agent completed
                output = ""
                if run.output:
                    # Handle ACP output format - run.output is already a list of messages
                    output_text = ""
                    for message in run.output:
                        if isinstance(message, dict) and "parts" in message:
                            for part in message["parts"]:
                                if isinstance(part, dict) and "content" in part:
                                    output_text += part["content"] + "\n"
                    output = output_text.strip() if output_text else "No text content"
                
                # Store final result
                self.interaction_results[run_id] = {
                    "output": output,
                    "error": run.error,
                    "completed_at": current_time
                }
                
                return {
                    "status": "completed",
                    "run_id": run.run_id,
                    "output": output,
                    "error": run.error
                }
                
        except Exception as e:
            # Clean up on error
            if run_id in self.pending_interactions:
                del self.pending_interactions[run_id]
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_pending_interactions(self) -> Dict[str, Any]:
        """Get all pending interactions"""
        
        current_time = asyncio.get_event_loop().time()
        active_interactions = {}
        expired_interactions = []
        
        for run_id, pending in self.pending_interactions.items():
            if current_time - pending.timestamp > pending.timeout_seconds:
                expired_interactions.append(run_id)
            else:
                active_interactions[run_id] = {
                    "agent_name": pending.agent_name,
                    "message": pending.await_message,
                    "waiting_seconds": int(current_time - pending.timestamp),
                    "timeout_in_seconds": int(pending.timeout_seconds - (current_time - pending.timestamp))
                }
        
        # Clean up expired interactions
        for run_id in expired_interactions:
            del self.pending_interactions[run_id]
        
        return {
            "active_count": len(active_interactions),
            "expired_count": len(expired_interactions),
            "interactions": active_interactions
        }
    
    async def cancel_interaction(self, run_id: str) -> bool:
        """Cancel a pending interaction"""
        
        if run_id in self.pending_interactions:
            del self.pending_interactions[run_id]
            return True
        return False

# Integration with FastMCP
def register_interactive_tools(mcp: FastMCP, manager: InteractiveManager):
    
    @mcp.tool()
    async def start_interactive_agent(
        agent_name: str,
        initial_input: str,
        session_id: str = None,
        timeout_minutes: int = 5
    ) -> str:
        """Start an interactive ACP agent that may require user input"""
        
        try:
            result = await manager.start_interactive_agent(
                agent_name=agent_name,
                initial_input=initial_input,
                session_id=session_id,
                timeout_seconds=timeout_minutes * 60
            )
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def provide_user_input(
        run_id: str,
        user_input: str
    ) -> str:
        """Provide user input to resume a waiting interactive agent"""
        
        try:
            result = await manager.resume_interactive_agent(run_id, user_input)
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def list_pending_interactions() -> str:
        """List all pending interactive agents waiting for input"""
        
        try:
            interactions = await manager.get_pending_interactions()
            
            return json.dumps(interactions, indent=2)
            
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def cancel_interaction(run_id: str) -> str:
        """Cancel a pending interactive agent"""
        
        try:
            success = await manager.cancel_interaction(run_id)
            
            if success:
                return f"Successfully cancelled interaction: {run_id}"
            else:
                return f"No pending interaction found with ID: {run_id}"
                
        except Exception as e:
            return f"Error: {e}"

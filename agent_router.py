# agent_router.py
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from fastmcp import FastMCP
from agent_discovery import AgentDiscoveryTool, ACPAgent
from run_orchestrator import RunOrchestrator

class RoutingRule(BaseModel):
    keywords: List[str]
    agent_name: str
    priority: int = 1
    description: str = ""

class RouterStrategy(BaseModel):
    name: str
    description: str
    rules: List[RoutingRule]

class AgentRouter:
    def __init__(
        self, 
        discovery: AgentDiscoveryTool,
        orchestrator: RunOrchestrator
    ):
        self.discovery = discovery
        self.orchestrator = orchestrator
        self.strategies: Dict[str, RouterStrategy] = {}
        self.default_strategy = self._create_default_strategy()
    
    def _create_default_strategy(self) -> RouterStrategy:
        """Create a default routing strategy based on common patterns"""
        return RouterStrategy(
            name="default",
            description="Default routing based on common keywords",
            rules=[
                RoutingRule(
                    keywords=["translate", "translation", "spanish", "french", "language"],
                    agent_name="translation",
                    priority=10,
                    description="Route translation requests"
                ),
                RoutingRule(
                    keywords=["weather", "temperature", "forecast", "climate"],
                    agent_name="weather",
                    priority=10,
                    description="Route weather requests"
                ),
                RoutingRule(
                    keywords=["calculate", "math", "compute", "sum", "multiply"],
                    agent_name="calculator",
                    priority=10,
                    description="Route calculation requests"
                ),
                RoutingRule(
                    keywords=["echo", "repeat", "test"],
                    agent_name="echo",
                    priority=5,
                    description="Route test requests to echo"
                )
            ]
        )
    
    def add_strategy(self, strategy: RouterStrategy):
        """Add a custom routing strategy"""
        self.strategies[strategy.name] = strategy
    
    async def route_request(
        self,
        input_text: str,
        strategy_name: str = "default",
        fallback_agent: str = "echo"
    ) -> str:
        """Route a request to the most appropriate agent"""
        
        # Get available agents
        available_agents = await self.discovery.discover_agents()
        available_agent_names = {agent.name for agent in available_agents}
        
        # Select strategy
        strategy = self.strategies.get(strategy_name, self.default_strategy)
        
        # Find best matching rule
        best_rule = None
        best_score = 0
        input_lower = input_text.lower()
        
        for rule in strategy.rules:
            # Check if agent is available
            if rule.agent_name not in available_agent_names:
                continue
                
            # Calculate match score
            score = 0
            for keyword in rule.keywords:
                if keyword.lower() in input_lower:
                    score += rule.priority
            
            if score > best_score:
                best_score = score
                best_rule = rule
        
        # Determine target agent
        if best_rule and best_score > 0:
            target_agent = best_rule.agent_name
            routing_reason = f"Matched rule: {best_rule.description} (score: {best_score})"
        else:
            # Fallback to first available agent or specified fallback
            if fallback_agent in available_agent_names:
                target_agent = fallback_agent
                routing_reason = f"Used fallback agent: {fallback_agent}"
            elif available_agents:
                target_agent = available_agents[0].name
                routing_reason = f"Used first available agent: {target_agent}"
            else:
                raise Exception("No agents available for routing")
        
        return target_agent, routing_reason
    
    async def execute_routed_request(
        self,
        input_text: str,
        strategy_name: str = "default",
        mode: str = "sync",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Route and execute a request"""
        
        try:
            # Route the request
            target_agent, routing_reason = await self.route_request(input_text, strategy_name)
            
            # Execute the agent
            if mode == "sync":
                run = await self.orchestrator.execute_agent_sync(
                    target_agent, 
                    input_text, 
                    session_id
                )
                
                result = {
                    "routed_to": target_agent,
                    "routing_reason": routing_reason,
                    "execution_mode": mode,
                    "status": run.status,
                    "run_id": run.run_id
                }
                
                if run.output:
                    # Convert output for display
                    from message_bridge import ACPMessage
                    mcp_content = await self.orchestrator.message_bridge.acp_to_mcp([
                        ACPMessage(parts=run.output)
                    ])
                    result["output"] = "\n".join([content.text for content in mcp_content if content.text])
                
                if run.error:
                    result["error"] = run.error
                
                return result
                
            else:
                # Async mode
                run_id = await self.orchestrator.execute_agent_async(
                    target_agent,
                    input_text,
                    session_id
                )
                
                return {
                    "routed_to": target_agent,
                    "routing_reason": routing_reason,
                    "execution_mode": mode,
                    "run_id": run_id,
                    "status": "async_started"
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "routed_to": None,
                "routing_reason": "Routing failed"
            }

# Integration with FastMCP
def register_router_tools(mcp: FastMCP, router: AgentRouter):
    
    @mcp.tool()
    async def smart_route_request(
        input_text: str,
        strategy: str = "default",
        mode: str = "sync",
        session_id: str = None
    ) -> str:
        """Intelligently route a request to the best ACP agent"""
        
        try:
            result = await router.execute_routed_request(
                input_text=input_text,
                strategy_name=strategy,
                mode=mode,
                session_id=session_id
            )
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def add_routing_rule(
        strategy_name: str,
        keywords: str,  # Comma-separated
        agent_name: str,
        priority: int = 5,
        description: str = ""
    ) -> str:
        """Add a new routing rule to a strategy"""
        
        try:
            keyword_list = [k.strip() for k in keywords.split(",")]
            
            new_rule = RoutingRule(
                keywords=keyword_list,
                agent_name=agent_name,
                priority=priority,
                description=description or f"Route to {agent_name}"
            )
            
            # Get or create strategy
            if strategy_name in router.strategies:
                strategy = router.strategies[strategy_name]
                strategy.rules.append(new_rule)
            else:
                strategy = RouterStrategy(
                    name=strategy_name,
                    description=f"Custom strategy: {strategy_name}",
                    rules=[new_rule]
                )
                router.strategies[strategy_name] = strategy
            
            return f"Successfully added rule: {keyword_list} -> {agent_name}"
            
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def list_routing_strategies() -> str:
        """List all available routing strategies and their rules"""
        
        try:
            strategies_info = {}
            
            # Include default strategy
            strategies_info["default"] = {
                "description": router.default_strategy.description,
                "rules": [
                    {
                        "keywords": rule.keywords,
                        "agent": rule.agent_name,
                        "priority": rule.priority,
                        "description": rule.description
                    }
                    for rule in router.default_strategy.rules
                ]
            }
            
            # Include custom strategies
            for name, strategy in router.strategies.items():
                strategies_info[name] = {
                    "description": strategy.description,
                    "rules": [
                        {
                            "keywords": rule.keywords,
                            "agent": rule.agent_name,
                            "priority": rule.priority,
                            "description": rule.description
                        }
                        for rule in strategy.rules
                    ]
                }
            
            return json.dumps(strategies_info, indent=2)
            
        except Exception as e:
            return f"Error: {e}"
    
    @mcp.tool()
    async def test_routing(
        input_text: str,
        strategy: str = "default"
    ) -> str:
        """Test routing without executing - shows which agent would be selected"""
        
        try:
            target_agent, routing_reason = await router.route_request(input_text, strategy)
            
            result = {
                "input": input_text,
                "strategy": strategy,
                "target_agent": target_agent,
                "routing_reason": routing_reason,
                "would_execute": True
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error: {e}"

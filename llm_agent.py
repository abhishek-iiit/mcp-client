import os
import logging
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, AIMessage
from typing import List, Dict, Any
import asyncio
from mcp_client import MCPClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

class MCPTool(BaseTool):
    """Wrapper to convert MCP tools to LangChain tools"""
    
    def __init__(self, name: str, description: str, mcp_client: MCPClient):
        super().__init__(name=name, description=description)
        object.__setattr__(self, 'mcp_client', mcp_client)
    
    def _run(self, **kwargs) -> str:
        """Execute the MCP tool"""
        logging.info(f"Calling MCP tool '{self.name}' with args: {kwargs}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Clean up arguments and replace placeholders with stored values
            clean_args = {}
            for k, v in kwargs.items():
                if isinstance(v, str) and v.endswith('_placeholder'):
                    # Extract key name from placeholder (e.g., 'session_id_placeholder' -> 'session_id')
                    key_name = v.replace('_placeholder', '')
                    if hasattr(self.mcp_client, 'agent_instance') and self.mcp_client.agent_instance:
                        stored_value = self.mcp_client.agent_instance.memory_context.get(key_name)
                        clean_args[k] = stored_value if stored_value else v
                    else:
                        clean_args[k] = v
                elif not (isinstance(v, str) and v.startswith('{') and v.endswith('}')):
                    clean_args[k] = v
            logging.info(f"Cleaned args: {clean_args}")
            result = loop.run_until_complete(
                self.mcp_client.call_tool(self.name, clean_args)
            )
            logging.info(f"MCP tool result: {result}")
            
            # Store response data in agent's memory
            if hasattr(self.mcp_client, 'agent_instance') and self.mcp_client.agent_instance:
                self.mcp_client.agent_instance._store_response_data(self.name, result)
            
            return str(result)
        finally:
            loop.close()

class LLMAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            temperature=0.0, 
            model=os.getenv("AZURE_OPENAI_MODEL"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )
        self.mcp_client = MCPClient(os.getenv("MCP_SERVER_URL"))
        self.mcp_client.agent_instance = self  # Reference for tools to access agent
        self.agent = None
        self.tools = []
        self.conversation_history = []
        self.memory_context = {}
    
    async def initialize(self):
        """Initialize the agent with MCP tools"""
        # Get tools from MCP server
        mcp_tools = await self.mcp_client.connect_and_get_tools()
        logging.info(f"Found {len(mcp_tools)} MCP tools: {[t.get('name') for t in mcp_tools]}")
        
        # Convert MCP tools to LangChain tools
        self.tools = []
        for tool in mcp_tools:
            # Create tool description with parameter info
            desc = tool.get('description', '')
            if 'inputSchema' in tool and 'properties' in tool['inputSchema']:
                params = list(tool['inputSchema']['properties'].keys())
                desc += f" Parameters: {', '.join(params)}"
                logging.info(f"Tool '{tool.get('name')}' parameters: {params}")
            
            langchain_tool = MCPTool(
                name=tool.get('name', ''),
                description=desc,
                mcp_client=self.mcp_client
            )
            self.tools.append(langchain_tool)
        
        # Create the agent with system prompt from .env
        system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
        self.agent = create_react_agent(self.llm, self.tools, state_modifier=system_prompt)
        
        return len(self.tools)
    
    async def chat(self, message: str):
        """Chat with the agent with enhanced memory"""
        if not self.agent:
            return "Agent not initialized. Please initialize first."
        
        try:
            logging.info(f"User message: {message}")
            
            # Build context-aware message with memory
            context_message = self._build_context_message(message)
            
            # Convert conversation history to proper message format
            messages = []
            for role, content in self.conversation_history:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            
            # Add current message
            messages.append(HumanMessage(content=context_message))
            
            # Send to agent
            response = await self.agent.ainvoke({"messages": messages})
            
            # Get assistant response
            assistant_response = response["messages"][-1].content
            
            # Update memory and history
            self._update_memory(message, assistant_response)
            self.conversation_history.append(("user", message))
            self.conversation_history.append(("assistant", assistant_response))
            
            logging.info(f"Agent response: {assistant_response}")
            return assistant_response
        except Exception as e:
            logging.error(f"Chat error: {e}")
            return f"Error: {e}"
    
    def _build_context_message(self, message: str) -> str:
        """Build message with relevant context from memory"""
        if not self.memory_context:
            return message
        
        context_parts = []
        for key, value in self.memory_context.items():
            if isinstance(value, (str, int, float)):
                context_parts.append(f"{key}: {value}")
            elif isinstance(value, dict) and value:  # Include response data
                context_parts.append(f"{key}: {value}")
        
        if context_parts:
            context = ", ".join(context_parts)
            return f"Context: {context}. {message}"
        
        return message
    
    def _update_memory(self, user_message: str, assistant_response: str):
        """Extract and store loan flow specific information"""
        import re
        
        text = user_message + " " + assistant_response
        
        # Extract phone numbers
        phone_pattern = r'\b(?:\+91[\s-]?)?[6-9]\d{9}\b'
        phones = re.findall(phone_pattern, text)
        if phones:
            self.memory_context['phone'] = phones[-1].strip()
        
        # Extract PAN
        pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
        pans = re.findall(pan_pattern, text)
        if pans:
            self.memory_context['pan'] = pans[-1]
        
        # Extract lead ID
        lead_patterns = [r'lead[_\s]?id[:\s]*([A-Za-z0-9-]+)', r'leadId[:\s]*([A-Za-z0-9-]+)']
        for pattern in lead_patterns:
            matches = re.findall(pattern, assistant_response, re.IGNORECASE)
            if matches:
                self.memory_context['lead_id'] = matches[-1]
        
        # Extract credit score
        score_pattern = r'credit[_\s]?score[:\s]*(\d{3})'  
        scores = re.findall(score_pattern, assistant_response, re.IGNORECASE)
        if scores:
            self.memory_context['credit_score'] = scores[-1]
        
        logging.info(f"Updated memory: {self.memory_context}")
    
    def _store_response_data(self, tool_name: str, response: Any):
        """Store MCP tool response data in memory"""
        import json
        
        try:
            # Parse response if it's a string containing JSON
            if isinstance(response, dict) and 'content' in response:
                content = response['content']
                if isinstance(content, list) and content:
                    text_content = content[0].get('text', '')
                    if text_content:
                        data = json.loads(text_content)
                        
                        # Store all response data dynamically
                        if 'data' in data and isinstance(data['data'], dict):
                            response_data = data['data']
                            for key, value in response_data.items():
                                if value and value != 'NA':  # Store non-empty, non-NA values
                                    # Convert camelCase to snake_case for consistency
                                    snake_key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
                                    self.memory_context[snake_key] = value
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Ignore parsing errors
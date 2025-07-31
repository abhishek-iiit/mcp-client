import os
import logging
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
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
            # Clean up arguments - remove any placeholder values
            clean_args = {k: v for k, v in kwargs.items() if not (isinstance(v, str) and v.startswith('{') and v.endswith('}'))}
            logging.info(f"Cleaned args: {clean_args}")
            result = loop.run_until_complete(
                self.mcp_client.call_tool(self.name, clean_args)
            )
            logging.info(f"MCP tool result: {result}")
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
        self.agent = None
        self.tools = []
        self.conversation_history = []
    
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
        
        # Create the agent
        system_prompt = "You are a helpful assistant that can use MCP tools. When users provide information like mobile numbers or OTPs, extract the actual values and pass them as parameters to the tools. Remember previous conversation context including mobile numbers, session IDs, and OTPs mentioned earlier. Do not use placeholder values."
        self.agent = create_react_agent(self.llm, self.tools, state_modifier=system_prompt)
        
        return len(self.tools)
    
    async def chat(self, message: str):
        """Chat with the agent"""
        if not self.agent:
            return "Agent not initialized. Please initialize first."
        
        try:
            logging.info(f"User message: {message}")
            
            # Add user message to history
            self.conversation_history.append(("user", message))
            
            # Send full conversation history to agent
            response = await self.agent.ainvoke({"messages": self.conversation_history})
            
            # Get assistant response and add to history
            assistant_response = response["messages"][-1].content
            self.conversation_history.append(("assistant", assistant_response))
            
            logging.info(f"Agent response: {assistant_response}")
            return assistant_response
        except Exception as e:
            logging.error(f"Chat error: {e}")
            return f"Error: {e}"
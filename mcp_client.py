import asyncio
import httpx
import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.tools = []
    
    async def connect_and_get_tools(self):
        """Connect to MCP server and fetch available tools using MCP protocol"""
        try:
            async with httpx.AsyncClient() as client:
                # MCP list_tools request
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list"
                }
                
                response = await client.post(
                    self.server_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logging.info(f"MCP tools response: {json.dumps(data, indent=2)}")
                    if 'result' in data and 'tools' in data['result']:
                        self.tools = data['result']['tools']
                        return self.tools
                        
                return []
        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a specific tool on the MCP server using MCP protocol"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }
                
                logging.info(f"Sending MCP request: {json.dumps(payload, indent=2)}")
                
                response = await client.post(
                    self.server_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                logging.info(f"MCP response status: {response.status_code}")
                logging.info(f"MCP response body: {response.text}")
                
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data:
                        return data['result']
                    elif 'error' in data:
                        return {"error": data['error']}
                        
                return {"error": f"Tool call failed: {response.status_code}"}
        except Exception as e:
            logging.error(f"MCP call error: {e}")
            return {"error": f"Error calling tool: {e}"}
# MCP Server Tester

A clean UI to test your MCP server with LLM integration.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Click "Initialize Agent" to connect to your MCP server
2. Start chatting to test your MCP tools
3. The LLM will automatically use available MCP tools to answer your questions

## Configuration

Configuration is loaded from `.env` file:
- **MCP Server**: Set via `MCP_SERVER_URL`
- **LLM Model**: Set via `AZURE_OPENAI_MODEL`
- **Azure OpenAI**: Configure endpoint, API key, and deployment

The app will automatically discover and integrate all available tools from your MCP server.
Thanks for reading this far
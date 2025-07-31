import streamlit as st
import asyncio
import os
from llm_agent import LLMAgent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="MCP Server Tester", page_icon="🤖", layout="wide")

st.title("🤖 MCP Server Tester")
st.markdown("Test your MCP server with LLM integration")

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Sidebar for connection info
with st.sidebar:
    st.header("🔧 Configuration")
    st.info(f"**MCP Server:** {os.getenv('MCP_SERVER_URL')}")
    st.info(f"**Model:** {os.getenv('AZURE_OPENAI_MODEL')}")
    
    if st.button("🔄 Initialize Agent", type="primary"):
        with st.spinner("Connecting to MCP server..."):
            try:
                agent = LLMAgent()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tool_count = loop.run_until_complete(agent.initialize())
                loop.close()
                
                st.session_state.agent = agent
                st.session_state.initialized = True
                st.success(f"✅ Connected! Found {tool_count} tools")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
    
    # Display memory context if agent is initialized
    if st.session_state.initialized and st.session_state.agent:
        st.header("🧠 Memory Context")
        memory = st.session_state.agent.memory_context
        if memory:
            for key, value in memory.items():
                st.text(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.text("No stored context yet")
        
        if st.button("🗑️ Clear Memory"):
            st.session_state.agent.memory_context = {}
            st.success("Memory cleared!")
            st.rerun()

# Main chat interface
if st.session_state.initialized:
    st.success("🟢 Agent Ready")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask something about your MCP server..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(st.session_state.agent.chat(prompt))
                    loop.close()
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    st.warning("⚠️ Please initialize the agent first using the sidebar button")
    st.markdown("### 📋 Instructions:")
    st.markdown("1. Click **Initialize Agent** in the sidebar")
    st.markdown("2. Wait for connection to your MCP server")
    st.markdown("3. Start chatting to test your MCP tools!")

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.agent:
            st.session_state.agent.conversation_history = []
        st.rerun()
import asyncio
import json
import logging
import random
import string
import threading
import time
import uuid
import weakref
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any, Set

import websockets
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_id():
    """Generate a random ID"""
    return str(uuid.uuid4())

def generate_user_name():
    """Generate a random user name if none is provided"""
    colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Cyan", "Magenta", "Lime"]
    animals = ["Panda", "Tiger", "Elephant", "Dolphin", "Eagle", "Wolf", "Fox", "Owl", "Bear", "Lion"]
    return f"{random.choice(colors)} {random.choice(animals)}"

class CollaborationClient:
    """Client for the collaboration server"""
    
    def __init__(self, server_url="ws://localhost:5002"):
        self.server_url = server_url
        self.user_id = generate_id()
        self.user_name = generate_user_name()
        self.session_id = None
        self.websocket = None
        self.connected = False
        self.session_info = None
        
        # Callbacks for different message types
        self.callbacks = {
            "user_joined": [],
            "user_left": [],
            "session_info": [],
            "chat": [],
            "data_update": [],
            "visualization_update": [],
            "analysis_update": []
        }
        
        # Thread for background processing
        self.thread = None
        self.running = False
    
    def set_user_info(self, user_id=None, user_name=None):
        """Set user information"""
        if user_id:
            self.user_id = user_id
        if user_name:
            self.user_name = user_name
    
    def register_callback(self, message_type: str, callback: Callable):
        """Register a callback for a specific message type"""
        if message_type not in self.callbacks:
            self.callbacks[message_type] = []
        self.callbacks[message_type].append(callback)
    
    def remove_callback(self, message_type: str, callback: Callable):
        """Remove a callback for a specific message type"""
        if message_type in self.callbacks and callback in self.callbacks[message_type]:
            self.callbacks[message_type].remove(callback)
    
    async def _connect(self, session_id):
        """Connect to the collaboration server"""
        self.session_id = session_id
        url = f"{self.server_url}/ws/{session_id}/{self.user_id}/{self.user_name}"
        
        try:
            self.websocket = await websockets.connect(url)
            self.connected = True
            logger.info(f"Connected to collaboration server: {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to collaboration server: {e}")
            self.connected = False
            return False
    
    async def _disconnect(self):
        """Disconnect from the collaboration server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False
            logger.info("Disconnected from collaboration server")
    
    async def _send_message(self, message_type: str, content: dict):
        """Send a message to the collaboration server"""
        if not self.connected or not self.websocket:
            logger.warning("Cannot send message: not connected")
            return False
        
        message = {
            "id": generate_id(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.connected = False
            return False
    
    async def _receive_messages(self):
        """Receive and process messages from the collaboration server"""
        if not self.connected or not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("message_type")
                    
                    # Store session info
                    if message_type == "session_info":
                        self.session_info = data.get("content", {}).get("session")
                    
                    # Call callbacks for this message type
                    if message_type in self.callbacks:
                        for callback in self.callbacks[message_type]:
                            try:
                                callback(data)
                            except Exception as e:
                                logger.error(f"Error in callback for {message_type}: {e}")
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error in receive_messages: {e}")
            self.connected = False
    
    async def _background_task(self):
        """Background task to maintain connection and process messages"""
        while self.running:
            try:
                if not self.connected and self.session_id:
                    success = await self._connect(self.session_id)
                    if not success:
                        # Wait before retrying
                        await asyncio.sleep(5)
                        continue
                
                # Receive and process messages
                await self._receive_messages()
                
                # If we get here, the connection was closed
                if self.running and self.session_id:
                    logger.info("Reconnecting...")
                    await asyncio.sleep(1)  # Wait before reconnecting
            except Exception as e:
                logger.error(f"Error in background task: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def _run_background_loop(self):
        """Run the event loop in a background thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._background_task())
        finally:
            loop.close()
    
    def start(self, session_id=None, user_name=None):
        """Start the collaboration client"""
        if user_name:
            self.user_name = user_name
        
        if not session_id:
            # Generate a new session ID if none is provided
            session_id = generate_id()
        
        self.session_id = session_id
        
        if self.running:
            return self.session_id
        
        self.running = True
        self.thread = threading.Thread(target=self._run_background_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Collaboration client started with session_id: {session_id}")
        return session_id
    
    def stop(self):
        """Stop the collaboration client"""
        self.running = False
        
        # Wait for the thread to exit
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        
        self.thread = None
        logger.info("Collaboration client stopped")
    
    def send_chat_message(self, text: str):
        """Send a chat message"""
        # Use a new thread to avoid blocking
        def _send():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._send_message("chat", {"text": text}))
            finally:
                loop.close()
        
        threading.Thread(target=_send).start()
    
    def send_data_update(self, data_type: str, data: Any):
        """Send a data update"""
        # Use a new thread to avoid blocking
        def _send():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._send_message("data_update", {
                    "type": data_type,
                    "data": data
                }))
            finally:
                loop.close()
        
        threading.Thread(target=_send).start()
    
    def send_visualization_update(self, chart_type: str, chart_data: Any, chart_options: dict = None):
        """Send a visualization update"""
        # Use a new thread to avoid blocking
        def _send():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._send_message("visualization_update", {
                    "type": chart_type,
                    "data": chart_data,
                    "options": chart_options or {}
                }))
            finally:
                loop.close()
        
        threading.Thread(target=_send).start()
    
    def send_analysis_update(self, analysis_type: str, parameters: dict, results: Any):
        """Send an analysis update"""
        # Use a new thread to avoid blocking
        def _send():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._send_message("analysis_update", {
                    "type": analysis_type,
                    "parameters": parameters,
                    "results": results
                }))
            finally:
                loop.close()
        
        threading.Thread(target=_send).start()
    
    def get_session_info(self):
        """Get information about the current session"""
        return self.session_info
    
    def get_users(self):
        """Get list of users in the current session"""
        if self.session_info and "users" in self.session_info:
            return self.session_info["users"]
        return []
    
    def is_connected(self):
        """Check if connected to the collaboration server"""
        return self.connected


# Streamlit integration
class StreamlitCollaboration:
    """Streamlit integration for the collaboration client"""
    
    def __init__(self, server_url="ws://localhost:5002"):
        self.client = CollaborationClient(server_url)
        
        # Initialize session state variables
        if "collab_initialized" not in st.session_state:
            st.session_state.collab_initialized = False
            st.session_state.collab_session_id = None
            st.session_state.collab_user_name = None
            st.session_state.collab_messages = []
            st.session_state.collab_users = []
    
    def initialize(self, session_id=None, user_name=None):
        """Initialize the collaboration client"""
        if st.session_state.collab_initialized:
            return
        
        # Use stored values or new ones
        session_id = session_id or st.session_state.collab_session_id
        user_name = user_name or st.session_state.collab_user_name or generate_user_name()
        
        # Start the client
        actual_session_id = self.client.start(session_id, user_name)
        
        # Store values in session state
        st.session_state.collab_session_id = actual_session_id
        st.session_state.collab_user_name = user_name
        st.session_state.collab_initialized = True
        
        # Initialize data sharing related session state variables
        if "shared_datasets" not in st.session_state:
            st.session_state.shared_datasets = {}
        if "shared_visualizations" not in st.session_state:
            st.session_state.shared_visualizations = {}
        if "shared_analyses" not in st.session_state:
            st.session_state.shared_analyses = {}
        
        # Register callbacks
        self.client.register_callback("chat", self._chat_callback)
        self.client.register_callback("user_joined", self._user_joined_callback)
        self.client.register_callback("user_left", self._user_left_callback)
        self.client.register_callback("session_info", self._session_info_callback)
        self.client.register_callback("data_update", self._data_update_callback)
        self.client.register_callback("visualization_update", self._visualization_update_callback)
        self.client.register_callback("analysis_update", self._analysis_update_callback)
        
        logger.info(f"Streamlit collaboration initialized with session: {actual_session_id}")
        return actual_session_id
    
    def cleanup(self):
        """Clean up the collaboration client"""
        if not st.session_state.collab_initialized:
            return
        
        self.client.stop()
        st.session_state.collab_initialized = False
    
    def _chat_callback(self, data):
        """Callback for chat messages"""
        message = {
            "user_id": data.get("user_id"),
            "user_name": self._get_user_name(data.get("user_id")),
            "text": data.get("content", {}).get("text", ""),
            "timestamp": data.get("timestamp")
        }
        st.session_state.collab_messages.append(message)
    
    def _user_joined_callback(self, data):
        """Callback for user joined events"""
        user = data.get("content", {}).get("user", {})
        
        # Add message to chat
        message = {
            "user_id": "system",
            "user_name": "System",
            "text": f"{user.get('name')} joined the session",
            "timestamp": data.get("timestamp")
        }
        st.session_state.collab_messages.append(message)
    
    def _user_left_callback(self, data):
        """Callback for user left events"""
        user_id = data.get("content", {}).get("user_id")
        user_name = self._get_user_name(user_id)
        
        # Add message to chat
        message = {
            "user_id": "system",
            "user_name": "System",
            "text": f"{user_name} left the session",
            "timestamp": data.get("timestamp")
        }
        st.session_state.collab_messages.append(message)
    
    def _session_info_callback(self, data):
        """Callback for session info updates"""
        session = data.get("content", {}).get("session", {})
        if "users" in session:
            st.session_state.collab_users = session["users"]
    
    def _data_update_callback(self, data):
        """Callback for data update messages"""
        content = data.get("content", {})
        user_id = data.get("user_id")
        user_name = self._get_user_name(user_id)
        data_type = content.get("type", "unknown")
        
        # Add a system message about the data update
        message = {
            "user_id": "system",
            "user_name": "System",
            "text": f"{user_name} shared {data_type} data",
            "timestamp": data.get("timestamp")
        }
        st.session_state.collab_messages.append(message)
        
        # Process the data update based on type
        if data_type == "dataset":
            # Handle dataset sharing
            dataset_info = content.get("data", {})
            
            if dataset_info:
                # Create a unique ID for this dataset
                dataset_id = str(uuid.uuid4())
                
                # Initialize shared_datasets if it doesn't exist
                if "shared_datasets" not in st.session_state:
                    st.session_state.shared_datasets = {}
                
                # Convert the shared data back to a DataFrame if possible
                try:
                    if "data" in dataset_info:
                        import pandas as pd
                        df = pd.DataFrame(dataset_info["data"])
                    elif "sample" in dataset_info:
                        # This is a sample of a larger dataset
                        import pandas as pd
                        df = pd.DataFrame(dataset_info["sample"])
                    else:
                        df = None
                        
                    # Store the dataset in session state
                    st.session_state.shared_datasets[dataset_id] = {
                        "metadata": {
                            "id": dataset_id,
                            "name": dataset_info.get("name", "Unnamed Dataset"),
                            "description": dataset_info.get("description", ""),
                            "timestamp": dataset_info.get("timestamp", datetime.now().isoformat()),
                            "shared_by": user_name
                        },
                        "data": df
                    }
                except Exception as e:
                    logger.error(f"Error processing shared dataset: {e}")
        
        # Handle other data types as needed
    
    def _visualization_update_callback(self, data):
        """Callback for visualization update messages"""
        content = data.get("content", {})
        user_id = data.get("user_id")
        user_name = self._get_user_name(user_id)
        chart_type = content.get("type", "unknown")
        chart_data = content.get("data", {})
        chart_options = content.get("options", {})
        
        # Add a system message about the visualization update
        message = {
            "user_id": "system",
            "user_name": "System",
            "text": f"{user_name} shared a {chart_type} visualization",
            "timestamp": data.get("timestamp")
        }
        st.session_state.collab_messages.append(message)
        
        # Store the visualization in session state
        viz_id = str(uuid.uuid4())
        
        # Initialize shared_visualizations if it doesn't exist
        if "shared_visualizations" not in st.session_state:
            st.session_state.shared_visualizations = {}
        
        st.session_state.shared_visualizations[viz_id] = {
            "metadata": {
                "id": viz_id,
                "type": chart_type,
                "title": chart_data.get("title", f"{chart_type.capitalize()} Visualization"),
                "timestamp": datetime.now().isoformat(),
                "shared_by": user_name
            },
            "data": {
                "type": chart_type,
                "data": chart_data,
                "options": chart_options
            }
        }
    
    def _analysis_update_callback(self, data):
        """Callback for analysis update messages"""
        content = data.get("content", {})
        user_id = data.get("user_id")
        user_name = self._get_user_name(user_id)
        analysis_type = content.get("type", "unknown")
        parameters = content.get("parameters", {})
        results = content.get("results", {})
        
        # Add a system message about the analysis update
        message = {
            "user_id": "system",
            "user_name": "System",
            "text": f"{user_name} shared {analysis_type} analysis results",
            "timestamp": data.get("timestamp")
        }
        st.session_state.collab_messages.append(message)
        
        # Store the analysis in session state
        analysis_id = str(uuid.uuid4())
        
        # Initialize shared_analyses if it doesn't exist
        if "shared_analyses" not in st.session_state:
            st.session_state.shared_analyses = {}
        
        st.session_state.shared_analyses[analysis_id] = {
            "metadata": {
                "id": analysis_id,
                "type": analysis_type,
                "dataset_name": parameters.get("dataset", "Unknown"),
                "timestamp": datetime.now().isoformat(),
                "shared_by": user_name
            },
            "data": results
        }
    
    def _get_user_name(self, user_id):
        """Get user name from user ID"""
        if user_id == "system":
            return "System"
        
        if user_id == self.client.user_id:
            return self.client.user_name
        
        for user in st.session_state.collab_users:
            if user.get("id") == user_id:
                return user.get("name")
        
        return "Unknown User"
    
    def render_collaboration_ui(self):
        """Render the collaboration UI in Streamlit"""
        if not st.session_state.collab_initialized:
            st.warning("Collaboration is not initialized. Call initialize() first.")
            return
        
        # Session information
        st.subheader("Collaborative Session")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"Session ID: `{st.session_state.collab_session_id}`")
            st.write(f"Your name: {st.session_state.collab_user_name}")
            
            # Copy session ID button
            if st.button("Copy Session ID"):
                st.write("Session ID copied to clipboard!")
        
        with col2:
            # Users in session
            st.write("Users in session:")
            for user in self.client.get_users():
                st.write(f"- {user.get('name')}")
        
        # Chat
        st.subheader("Chat")
        
        # Display messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.collab_messages[-10:]:  # Show last 10 messages
                if message.get("user_id") == "system":
                    st.info(message.get("text"))
                elif message.get("user_id") == self.client.user_id:
                    st.write(f"**You**: {message.get('text')}")
                else:
                    st.write(f"**{message.get('user_name')}**: {message.get('text')}")
        
        # Chat input
        chat_input = st.text_input("Type a message")
        if st.button("Send") and chat_input:
            self.client.send_chat_message(chat_input)
            st.experimental_rerun()
    
    def share_data(self, data_type: str, data: Any):
        """Share data with other users"""
        if not st.session_state.collab_initialized:
            logger.warning("Collaboration is not initialized. Cannot share data.")
            return
        
        self.client.send_data_update(data_type, data)
    
    def share_visualization(self, chart_type: str, chart_data: Any, chart_options: dict = None):
        """Share a visualization with other users"""
        if not st.session_state.collab_initialized:
            logger.warning("Collaboration is not initialized. Cannot share visualization.")
            return
        
        self.client.send_visualization_update(chart_type, chart_data, chart_options)
    
    def share_analysis(self, analysis_type: str, parameters: dict, results: Any):
        """Share analysis results with other users"""
        if not st.session_state.collab_initialized:
            logger.warning("Collaboration is not initialized. Cannot share analysis.")
            return
        
        self.client.send_analysis_update(analysis_type, parameters, results)
    
    def collaboration_sidebar(self):
        """Render collaboration controls in the sidebar"""
        st.sidebar.header("Collaboration")
        
        if not st.session_state.collab_initialized:
            # Join existing session or create new one
            join_option = st.sidebar.radio("Collaboration Options:", 
                                          ["Create New Session", "Join Existing Session"])
            
            user_name = st.sidebar.text_input("Your Name:", 
                                              value=generate_user_name())
            
            if join_option == "Join Existing Session":
                session_id = st.sidebar.text_input("Session ID:")
                if st.sidebar.button("Join Session") and session_id:
                    self.initialize(session_id=session_id, user_name=user_name)
                    st.experimental_rerun()
            else:
                if st.sidebar.button("Create Session"):
                    self.initialize(user_name=user_name)
                    st.experimental_rerun()
        else:
            # Already in a session
            st.sidebar.success(f"Connected to session: {st.session_state.collab_session_id[:8]}...")
            st.sidebar.write(f"Your name: {st.session_state.collab_user_name}")
            
            # Number of users
            user_count = len(self.client.get_users())
            st.sidebar.write(f"Users in session: {user_count}")
            
            if st.sidebar.button("Leave Session"):
                self.cleanup()
                st.experimental_rerun()
            
            # Share current view
            st.sidebar.subheader("Share Current View")
            if st.sidebar.button("Share Current Analysis"):
                # This would need to be implemented by the main app
                st.sidebar.success("Analysis shared with collaborators!")


# Example usage
if __name__ == "__main__":
    import streamlit as st
    
    st.title("Collaboration Example")
    
    # Initialize collaboration
    collab = StreamlitCollaboration()
    
    # Show collaboration sidebar
    collab.collaboration_sidebar()
    
    if "collab_initialized" in st.session_state and st.session_state.collab_initialized:
        # Show collaboration UI
        collab.render_collaboration_ui()
        
        # Example of sharing data
        if st.button("Share Example Data"):
            data = {"example": "data", "value": 123}
            collab.share_data("example", data)
            st.success("Data shared!")
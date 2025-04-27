import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Set, Optional

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data models
class User(BaseModel):
    id: str
    name: str
    session_id: str
    connected_at: datetime = Field(default_factory=datetime.now)
    
class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    message_type: str
    content: dict
    timestamp: datetime = Field(default_factory=datetime.now)

class Session(BaseModel):
    id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    owner_id: str
    users: Dict[str, User] = {}
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "owner_id": self.owner_id,
            "user_count": len(self.users),
            "users": [user.dict() for user in self.users.values()]
        }

# Collaboration manager
class CollaborationManager:
    def __init__(self):
        self.active_sessions: Dict[str, Session] = {}
        self.connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str, session_id: str, user_name: str):
        await websocket.accept()
        
        # Create a new session if it doesn't exist
        if session_id not in self.active_sessions:
            session = Session(id=session_id, name=f"Session {session_id[:8]}", owner_id=user_id)
            self.active_sessions[session_id] = session
            logger.info(f"Created new session: {session_id}")
        
        # Add user to session
        user = User(id=user_id, name=user_name, session_id=session_id)
        self.active_sessions[session_id].users[user_id] = user
        
        # Store connection
        self.connections[user_id] = websocket
        
        # Notify everyone in the session about the new user
        await self.broadcast_to_session(
            session_id=session_id,
            message=Message(
                user_id=user_id,
                session_id=session_id,
                message_type="user_joined",
                content={"user": user.dict()}
            )
        )
        
        # Send session info to the new user
        await self.send_personal_message(
            user_id=user_id,
            message=Message(
                user_id="system",
                session_id=session_id,
                message_type="session_info",
                content={"session": self.active_sessions[session_id].to_dict()}
            )
        )
        
        logger.info(f"User {user_id} ({user_name}) connected to session {session_id}")
    
    async def disconnect(self, user_id: str):
        # Find user's session
        session_id = None
        for s_id, session in self.active_sessions.items():
            if user_id in session.users:
                session_id = s_id
                break
        
        if not session_id:
            logger.warning(f"User {user_id} disconnected but was not in any session")
            return
        
        # Remove user from session
        user = self.active_sessions[session_id].users.pop(user_id, None)
        
        # Remove connection
        self.connections.pop(user_id, None)
        
        # If session is empty, remove it
        if not self.active_sessions[session_id].users:
            self.active_sessions.pop(session_id, None)
            logger.info(f"Session {session_id} removed (no users)")
        else:
            # Notify everyone in the session about the user leaving
            await self.broadcast_to_session(
                session_id=session_id,
                message=Message(
                    user_id="system",
                    session_id=session_id,
                    message_type="user_left",
                    content={"user_id": user_id}
                )
            )
        
        logger.info(f"User {user_id} disconnected from session {session_id}")
    
    async def broadcast_to_session(self, session_id: str, message: Message):
        """Send a message to all users in a session"""
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to broadcast to non-existent session {session_id}")
            return
        
        serialized_message = message.json()
        
        # Get all users in the session
        users = self.active_sessions[session_id].users
        
        # Send to each connected user
        for user_id in users:
            if user_id in self.connections:
                try:
                    await self.connections[user_id].send_text(serialized_message)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
    
    async def send_personal_message(self, user_id: str, message: Message):
        """Send a message to a specific user"""
        if user_id not in self.connections:
            logger.warning(f"Attempted to send message to non-connected user {user_id}")
            return
        
        try:
            await self.connections[user_id].send_text(message.json())
        except Exception as e:
            logger.error(f"Error sending personal message to user {user_id}: {e}")
    
    async def process_message(self, user_id: str, message_data: dict):
        """Process an incoming message from a user"""
        try:
            message = Message(**message_data)
            
            # Make sure the message has the correct user_id
            message.user_id = user_id
            
            # Check if the session exists
            if message.session_id not in self.active_sessions:
                logger.warning(f"Message for non-existent session {message.session_id}")
                return
            
            # Check if the user is in the session
            if user_id not in self.active_sessions[message.session_id].users:
                logger.warning(f"Message from user {user_id} who is not in session {message.session_id}")
                return
            
            # Process message based on type
            if message.message_type == "chat":
                # Broadcast chat messages to all users in the session
                await self.broadcast_to_session(message.session_id, message)
            
            elif message.message_type == "data_update":
                # Broadcast data updates to all users in the session
                await self.broadcast_to_session(message.session_id, message)
            
            elif message.message_type == "visualization_update":
                # Broadcast visualization updates to all users in the session
                await self.broadcast_to_session(message.session_id, message)
            
            elif message.message_type == "analysis_update":
                # Broadcast analysis updates to all users in the session
                await self.broadcast_to_session(message.session_id, message)
            
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def get_active_sessions(self):
        """Return list of active sessions"""
        return [session.to_dict() for session in self.active_sessions.values()]
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a specific session by ID"""
        return self.active_sessions.get(session_id)

# Create FastAPI app
app = FastAPI(title="Data Mining Collaboration Server")

# Create collaboration manager
manager = CollaborationManager()

@app.get("/")
async def root():
    return {"status": "running", "sessions": len(manager.active_sessions)}

@app.get("/sessions")
async def get_sessions():
    return manager.get_active_sessions()

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = manager.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    return session.to_dict()

@app.websocket("/ws/{session_id}/{user_id}/{user_name}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, user_id: str, user_name: str):
    await manager.connect(websocket, user_id, session_id, user_name)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            await manager.process_message(user_id, message_data)
    except WebSocketDisconnect:
        await manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"Error in websocket connection: {e}")
        await manager.disconnect(user_id)

# Run the server using uvicorn when script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=5002, reload=True)
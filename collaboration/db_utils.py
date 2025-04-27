import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

# Import database manager
try:
    from utils.db_utils import DatabaseManager
except ImportError:
    # For standalone testing
    class DatabaseManager:
        def __init__(self):
            self.engine = create_engine('sqlite:///collaboration.db')
            self.Session = sessionmaker(bind=self.engine)
        
        def test_connection(self):
            return True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database connection
db_manager = DatabaseManager()
Base = declarative_base()

class CollaborationSession(Base):
    """Model for collaboration sessions"""
    __tablename__ = 'collaboration_sessions'
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    owner_id = Column(String(36), nullable=False)
    active = Column(Boolean, default=True)
    last_activity = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    users = relationship("CollaborationUser", back_populates="session", cascade="all, delete-orphan")
    messages = relationship("CollaborationMessage", back_populates="session", cascade="all, delete-orphan")
    shared_data = relationship("SharedData", back_populates="session", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "owner_id": self.owner_id,
            "active": self.active,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "user_count": len(self.users) if self.users else 0
        }

class CollaborationUser(Base):
    """Model for users in collaboration sessions"""
    __tablename__ = 'collaboration_users'
    
    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey('collaboration_sessions.id'), nullable=False)
    name = Column(String(255), nullable=False)
    connected_at = Column(DateTime, default=datetime.now)
    last_activity = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    session = relationship("CollaborationSession", back_populates="users")
    messages = relationship("CollaborationMessage", back_populates="user", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "is_active": self.is_active
        }

class CollaborationMessage(Base):
    """Model for messages in collaboration sessions"""
    __tablename__ = 'collaboration_messages'
    
    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey('collaboration_sessions.id'), nullable=False)
    user_id = Column(String(36), ForeignKey('collaboration_users.id'), nullable=False)
    message_type = Column(String(50), nullable=False)
    content = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    
    # Relationships
    session = relationship("CollaborationSession", back_populates="messages")
    user = relationship("CollaborationUser", back_populates="messages")
    
    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class SharedData(Base):
    """Model for shared data in collaboration sessions"""
    __tablename__ = 'shared_data'
    
    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey('collaboration_sessions.id'), nullable=False)
    user_id = Column(String(36), nullable=False)
    data_type = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    session = relationship("CollaborationSession", back_populates="shared_data")
    
    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "data_type": self.data_type,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class CollaborationDatabaseManager:
    """Database manager for collaboration features"""
    
    def __init__(self):
        """Initialize the database connection"""
        self.db_manager = db_manager
        self.engine = self.db_manager.engine
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        logger.info("Collaboration database tables created or verified")
    
    def create_session(self, session_id: str, name: str, owner_id: str) -> str:
        """Create a new collaboration session"""
        session = self.Session()
        try:
            collab_session = CollaborationSession(
                id=session_id,
                name=name,
                owner_id=owner_id,
                active=True
            )
            session.add(collab_session)
            session.commit()
            logger.info(f"Created collaboration session: {session_id}")
            return session_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating collaboration session: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a collaboration session by ID"""
        session = self.Session()
        try:
            collab_session = session.query(CollaborationSession).filter_by(id=session_id).first()
            if collab_session:
                return collab_session.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting collaboration session: {e}")
            return None
        finally:
            session.close()
    
    def get_active_sessions(self, limit: int = 100) -> List[dict]:
        """Get all active collaboration sessions"""
        session = self.Session()
        try:
            collab_sessions = session.query(CollaborationSession).filter_by(active=True).order_by(
                CollaborationSession.last_activity.desc()).limit(limit).all()
            return [s.to_dict() for s in collab_sessions]
        except Exception as e:
            logger.error(f"Error getting active collaboration sessions: {e}")
            return []
        finally:
            session.close()
    
    def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a collaboration session"""
        session = self.Session()
        try:
            collab_session = session.query(CollaborationSession).filter_by(id=session_id).first()
            if collab_session:
                collab_session.active = False
                session.commit()
                logger.info(f"Deactivated collaboration session: {session_id}")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deactivating collaboration session: {e}")
            return False
        finally:
            session.close()
    
    def add_user_to_session(self, session_id: str, user_id: str, user_name: str) -> bool:
        """Add a user to a collaboration session"""
        session = self.Session()
        try:
            # Check if user already exists in session
            existing_user = session.query(CollaborationUser).filter_by(
                session_id=session_id, id=user_id).first()
            
            if existing_user:
                # Update existing user
                existing_user.is_active = True
                existing_user.last_activity = datetime.now()
                existing_user.name = user_name  # Update name in case it changed
                session.commit()
                logger.info(f"Updated user {user_id} in session {session_id}")
                return True
            
            # Check if session exists
            collab_session = session.query(CollaborationSession).filter_by(id=session_id).first()
            if not collab_session:
                logger.warning(f"Session {session_id} not found")
                return False
            
            # Add new user
            user = CollaborationUser(
                id=user_id,
                session_id=session_id,
                name=user_name,
                is_active=True
            )
            session.add(user)
            
            # Update session last activity
            collab_session.last_activity = datetime.now()
            
            session.commit()
            logger.info(f"Added user {user_id} to session {session_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding user to session: {e}")
            return False
        finally:
            session.close()
    
    def remove_user_from_session(self, session_id: str, user_id: str) -> bool:
        """Mark a user as inactive in a collaboration session"""
        session = self.Session()
        try:
            user = session.query(CollaborationUser).filter_by(
                session_id=session_id, id=user_id).first()
            
            if user:
                user.is_active = False
                user.last_activity = datetime.now()
                
                # Update session last activity
                collab_session = session.query(CollaborationSession).filter_by(id=session_id).first()
                if collab_session:
                    collab_session.last_activity = datetime.now()
                
                session.commit()
                logger.info(f"Removed user {user_id} from session {session_id}")
                return True
            
            logger.warning(f"User {user_id} not found in session {session_id}")
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing user from session: {e}")
            return False
        finally:
            session.close()
    
    def get_session_users(self, session_id: str, active_only: bool = True) -> List[dict]:
        """Get all users in a collaboration session"""
        session = self.Session()
        try:
            query = session.query(CollaborationUser).filter_by(session_id=session_id)
            if active_only:
                query = query.filter_by(is_active=True)
            
            users = query.all()
            return [user.to_dict() for user in users]
        except Exception as e:
            logger.error(f"Error getting session users: {e}")
            return []
        finally:
            session.close()
    
    def add_message(self, session_id: str, user_id: str, message_type: str, content: dict) -> str:
        """Add a message to a collaboration session"""
        session = self.Session()
        try:
            message_id = str(uuid.uuid4())
            message = CollaborationMessage(
                id=message_id,
                session_id=session_id,
                user_id=user_id,
                message_type=message_type,
                content=content
            )
            session.add(message)
            
            # Update user and session last activity
            user = session.query(CollaborationUser).filter_by(
                session_id=session_id, id=user_id).first()
            if user:
                user.last_activity = datetime.now()
            
            collab_session = session.query(CollaborationSession).filter_by(id=session_id).first()
            if collab_session:
                collab_session.last_activity = datetime.now()
            
            session.commit()
            logger.info(f"Added message {message_id} to session {session_id}")
            return message_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding message to session: {e}")
            raise
        finally:
            session.close()
    
    def get_messages(self, session_id: str, limit: int = 100, message_type: str = None) -> List[dict]:
        """Get messages from a collaboration session"""
        session = self.Session()
        try:
            query = session.query(CollaborationMessage).filter_by(session_id=session_id)
            if message_type:
                query = query.filter_by(message_type=message_type)
            
            messages = query.order_by(CollaborationMessage.timestamp.desc()).limit(limit).all()
            return [message.to_dict() for message in messages]
        except Exception as e:
            logger.error(f"Error getting session messages: {e}")
            return []
        finally:
            session.close()
    
    def share_data(self, session_id: str, user_id: str, data_type: str, 
                  name: str, data: dict, description: str = None) -> str:
        """Share data in a collaboration session"""
        session = self.Session()
        try:
            data_id = str(uuid.uuid4())
            shared_data = SharedData(
                id=data_id,
                session_id=session_id,
                user_id=user_id,
                data_type=data_type,
                name=name,
                description=description,
                data=data
            )
            session.add(shared_data)
            
            # Update user and session last activity
            user = session.query(CollaborationUser).filter_by(
                session_id=session_id, id=user_id).first()
            if user:
                user.last_activity = datetime.now()
            
            collab_session = session.query(CollaborationSession).filter_by(id=session_id).first()
            if collab_session:
                collab_session.last_activity = datetime.now()
            
            session.commit()
            logger.info(f"Shared data {data_id} in session {session_id}")
            return data_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error sharing data in session: {e}")
            raise
        finally:
            session.close()
    
    def get_shared_data(self, session_id: str, data_type: str = None) -> List[dict]:
        """Get shared data from a collaboration session"""
        session = self.Session()
        try:
            query = session.query(SharedData).filter_by(session_id=session_id)
            if data_type:
                query = query.filter_by(data_type=data_type)
            
            shared_data = query.order_by(SharedData.created_at.desc()).all()
            return [data.to_dict() for data in shared_data]
        except Exception as e:
            logger.error(f"Error getting shared data from session: {e}")
            return []
        finally:
            session.close()
    
    def get_shared_data_by_id(self, data_id: str) -> Optional[dict]:
        """Get shared data by ID"""
        session = self.Session()
        try:
            data = session.query(SharedData).filter_by(id=data_id).first()
            if data:
                result = data.to_dict()
                result['data'] = data.data  # Include the actual data
                return result
            return None
        except Exception as e:
            logger.error(f"Error getting shared data by ID: {e}")
            return None
        finally:
            session.close()
    
    def cleanup_inactive_sessions(self, hours: int = 24) -> int:
        """Clean up inactive sessions older than the specified hours"""
        session = self.Session()
        try:
            cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
            inactive_sessions = session.query(CollaborationSession).filter(
                CollaborationSession.active == True,
                CollaborationSession.last_activity < cutoff_time
            ).all()
            
            count = 0
            for collab_session in inactive_sessions:
                collab_session.active = False
                count += 1
            
            session.commit()
            logger.info(f"Cleaned up {count} inactive sessions")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up inactive sessions: {e}")
            return 0
        finally:
            session.close()


# Singleton instance
collaboration_db = CollaborationDatabaseManager()


# Test the database connection
if __name__ == "__main__":
    try:
        # Create a test session
        session_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        
        collaboration_db.create_session(session_id, "Test Session", user_id)
        collaboration_db.add_user_to_session(session_id, user_id, "Test User")
        
        # Test getting the session
        session = collaboration_db.get_session(session_id)
        print(f"Created session: {session}")
        
        # Test adding a message
        message_id = collaboration_db.add_message(
            session_id, user_id, "chat", {"text": "Hello, world!"})
        
        # Test getting messages
        messages = collaboration_db.get_messages(session_id)
        print(f"Messages: {messages}")
        
        # Test sharing data
        data_id = collaboration_db.share_data(
            session_id, user_id, "test_data", "Test Data", 
            {"value": 42}, "This is a test")
        
        # Test getting shared data
        shared_data = collaboration_db.get_shared_data(session_id)
        print(f"Shared data: {shared_data}")
        
        print("All tests passed!")
    except Exception as e:
        print(f"Error testing database: {e}")
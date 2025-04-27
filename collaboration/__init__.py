"""
Collaboration Module for Data Mining and Weather Dashboard Applications

This module provides real-time collaboration features for data analysis and visualization,
allowing multiple users to work together on data exploration and share insights.

Components:
- WebSocket server for real-time communication
- Client library for browser-based interaction
- Database utilities for storing collaboration data
- Streamlit integration for easy UI rendering
"""

from collaboration.client import CollaborationClient, StreamlitCollaboration
from collaboration.db_utils import collaboration_db

__all__ = ['CollaborationClient', 'StreamlitCollaboration', 'collaboration_db']
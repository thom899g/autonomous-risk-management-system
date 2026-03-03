"""
Firebase Client Module
Handles all Firebase operations with connection pooling and error recovery.
Architectural Choice: Factory pattern with connection pooling ensures efficient
resource usage and automatic reconnection on failures.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

import firebase_admin
from firebase_admin import credentials, firestore, db
from firebase_admin.exceptions import FirebaseError

from .config import config

logger = logging.getLogger(__name__)

class FirebaseConnectionPool:
    """Manages Firebase connection pooling"""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self._connections = []
        self._lock = threading.Lock()
        self._
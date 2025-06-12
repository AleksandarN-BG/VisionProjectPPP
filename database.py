import sqlite3
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional


class ObjectDatabase:
    def __init__(self, db_path="objects.db"):
        """Initialize the database connection and create tables if they don't exist."""
        # Ensure the database directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            bbox_x INTEGER,
            bbox_y INTEGER,
            bbox_width INTEGER,
            bbox_height INTEGER,
            session_id TEXT
        )
        ''')
        self.conn.commit()
    
    def add_object(self, name: str, confidence: float, bbox: Optional[Dict] = None, session_id: Optional[str] = None) -> int:
        """
        Add a new detected object to the database.
        
        Args:
            name: The class name of the detected object
            confidence: The confidence score (0-1)
            bbox: Optional dictionary with bounding box coordinates (origin_x, origin_y, width, height)
            session_id: Optional session identifier to group detections
            
        Returns:
            The ID of the newly created record
        """
        timestamp = datetime.now().isoformat()
        
        # Extract bounding box data if provided
        bbox_x = bbox.get('origin_x') if bbox else None
        bbox_y = bbox.get('origin_y') if bbox else None
        bbox_width = bbox.get('width') if bbox else None
        bbox_height = bbox.get('height') if bbox else None
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        self.cursor.execute(
            """INSERT INTO detected_objects 
               (name, confidence, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, session_id) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, confidence, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, session_id)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_all_objects(self) -> List[Dict[str, Any]]:
        """
        Get all detected objects from the database.
        
        Returns:
            A list of dictionaries containing object data
        """
        self.cursor.execute("""
            SELECT id, name, confidence, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, session_id 
            FROM detected_objects
            ORDER BY timestamp DESC
        """)
        rows = self.cursor.fetchall()
        
        objects = []
        for row in rows:
            obj_id, name, confidence, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, session_id = row
            objects.append({
                "id": obj_id,
                "name": name,
                "confidence": confidence,
                "timestamp": timestamp,
                "bbox": {
                    "origin_x": bbox_x,
                    "origin_y": bbox_y,
                    "width": bbox_width,
                    "height": bbox_height
                } if bbox_x is not None else None,
                "session_id": session_id
            })
        
        return objects
    
    def get_objects_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all objects from a specific detection session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            A list of dictionaries containing object data
        """
        self.cursor.execute("""
            SELECT id, name, confidence, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, session_id 
            FROM detected_objects
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """, (session_id,))
        rows = self.cursor.fetchall()
        
        objects = []
        for row in rows:
            obj_id, name, confidence, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, session_id = row
            objects.append({
                "id": obj_id,
                "name": name,
                "confidence": confidence,
                "timestamp": timestamp,
                "bbox": {
                    "origin_x": bbox_x,
                    "origin_y": bbox_y,
                    "width": bbox_width,
                    "height": bbox_height
                } if bbox_x is not None else None,
                "session_id": session_id
            })
        
        return objects
    
    def get_object_by_id(self, object_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific object by its ID.
        
        Args:
            object_id: The ID of the object to retrieve
            
        Returns:
            A dictionary containing the object data, or None if not found
        """
        self.cursor.execute("""
            SELECT id, name, confidence, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, session_id 
            FROM detected_objects
            WHERE id = ?
        """, (object_id,))
        row = self.cursor.fetchone()
        
        if row:
            obj_id, name, confidence, timestamp, bbox_x, bbox_y, bbox_width, bbox_height, session_id = row
            return {
                "id": obj_id,
                "name": name,
                "confidence": confidence,
                "timestamp": timestamp,
                "bbox": {
                    "origin_x": bbox_x,
                    "origin_y": bbox_y,
                    "width": bbox_width,
                    "height": bbox_height
                } if bbox_x is not None else None,
                "session_id": session_id
            }
        return None
    
    def delete_object(self, object_id: int) -> bool:
        """
        Delete an object by its ID.
        
        Args:
            object_id: The ID of the object to delete
            
        Returns:
            True if the deletion was successful, False otherwise
        """
        self.cursor.execute("DELETE FROM detected_objects WHERE id = ?", (object_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def get_object_counts(self) -> Dict[str, int]:
        """
        Get counts of each object type detected.
        
        Returns:
            A dictionary with object names as keys and counts as values
        """
        self.cursor.execute("""
            SELECT name, COUNT(*) as count
            FROM detected_objects
            GROUP BY name
            ORDER BY count DESC
        """)
        rows = self.cursor.fetchall()
        
        return {name: count for name, count in rows}
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Ensure the database connection is closed when the object is deleted."""
        self.close()

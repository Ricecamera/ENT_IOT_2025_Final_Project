#!/usr/bin/env python3
"""
Face Database Management System
Handles storage, retrieval, and management of face embeddings and metadata.
"""

import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pytz

LOCAL_TZ = pytz.timezone('Asia/Bangkok')


class FaceDatabase:
    """
    Database for managing face recognition data including embeddings,
    metadata, and attendance logs.
    """
    
    def __init__(self, db_path="face_database"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.metadata_file = self.db_path / "metadata.json"
        self.embeddings_file = self.db_path / "embeddings.pkl"
        self.log_file = self.db_path / "checkin_log.json"
        
        self.metadata = {}
        self.embeddings = {}
        self.logs = []
        
        self.load()

    def load(self):
        """Load database files from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)

    def save(self):
        """Save database files to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def add_person(self, person_id, name, embedding, image_path=None):
        """
        Add a new person to the database.
        
        Args:
            person_id: Unique identifier for the person
            name: Person's name
            embedding: Face embedding vector (numpy array)
            image_path: Optional path to person's image
        
        Returns:
            True if successful
        """
        self.metadata[person_id] = {
            'name': name,
            'enrolled_at': datetime.now(LOCAL_TZ).isoformat(),
            'checkin_count': 0,
            'last_checkin': None,
            'image_path': str(image_path) if image_path else None
        }
        self.embeddings[person_id] = embedding
        self.save()
        return True

    def remove_person(self, person_id):
        """
        Remove a person from the database.
        
        Args:
            person_id: Unique identifier of the person to remove
        
        Returns:
            True if person was removed, False if person_id not found
        """
        if person_id not in self.metadata:
            return False
        
        # Remove from metadata and embeddings
        del self.metadata[person_id]
        if person_id in self.embeddings:
            del self.embeddings[person_id]
        
        # Remove from logs (optional - keeps history clean)
        # Uncomment if you want to remove all logs for this person
        # self.logs = [log for log in self.logs if log['person_id'] != person_id]
        
        self.save()
        return True

    def clear_database(self):
        """
        Clear all persons from the database (keeps logs).
        WARNING: This operation cannot be undone!
        
        Returns:
            Number of persons removed
        """
        count = len(self.metadata)
        self.metadata = {}
        self.embeddings = {}
        self.save()
        return count

    def record_checkin(self, person_id):
        """
        Record a check-in for a person.
        
        Args:
            person_id: Unique identifier of the person
        
        Returns:
            True if this is a new day check-in, False otherwise
        """
        if person_id not in self.metadata:
            return False
        
        now = datetime.now(LOCAL_TZ)
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        
        last_checkin_str = self.metadata[person_id].get('last_checkin')
        is_new_day = True
        
        if last_checkin_str and last_checkin_str.split(' ')[0] == current_date:
            is_new_day = False

        if is_new_day:
            self.metadata[person_id]['checkin_count'] = \
                self.metadata[person_id].get('checkin_count', 0) + 1
            
        self.logs.append({
            'person_id': person_id,
            'name': self.metadata[person_id]['name'],
            'timestamp': current_time,
            'is_counted': is_new_day
        })
        
        # Keep only recent logs to prevent file from growing too large
        if len(self.logs) > 5000:
            self.logs = self.logs[-5000:]
        
        self.metadata[person_id]['last_checkin'] = current_time
        self.save()
        return is_new_day

    def search(self, query_embedding, threshold=0.7, top_k=1):
        """
        Search for matching faces in the database.
        
        Args:
            query_embedding: Face embedding to search for
            threshold: Minimum similarity threshold (0-1)
            top_k: Number of top matches to return
        
        Returns:
            List of matching persons with similarity scores
        """
        if not self.embeddings:
            return []

        matches = []
        for person_id, db_embedding in self.embeddings.items():
            similarity = float(np.dot(query_embedding, db_embedding))
            
            if similarity >= threshold:
                meta = self.metadata.get(person_id, {})
                matches.append({
                    'person_id': person_id,
                    'name': meta.get('name', 'Unknown'),
                    'similarity': similarity,
                    'checkin_count': meta.get('checkin_count', 0),
                    'last_checkin': meta.get('last_checkin', 'Never'),
                })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:top_k]

    def get_statistics(self):
        """
        Get database statistics.
        
        Returns:
            Dictionary with various statistics
        """
        total_persons = len(self.metadata)
        total_logs = len(self.logs)
        total_checkins = sum(meta.get('checkin_count', 0) for meta in self.metadata.values())
        
        return {
            'total_persons': total_persons,
            'total_logs': total_logs,
            'total_checkins': total_checkins,
            'database_path': str(self.db_path)
        }

    def __len__(self):
        """Return number of persons in database."""
        return len(self.metadata)

    def __contains__(self, person_id):
        """Check if person exists in database."""
        return person_id in self.metadata

    def __repr__(self):
        return f"FaceDatabase(path='{self.db_path}', persons={len(self)})"

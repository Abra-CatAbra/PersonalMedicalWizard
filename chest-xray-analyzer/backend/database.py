"""
SQLite Database Integration for Medical Test Data Storage
Handles storage and retrieval of DICOM analysis results
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_PATH = "medical_test_data.db"

@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    id: Optional[int] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None  # 'DICOM' or 'Image'
    patient_id: Optional[str] = None
    study_date: Optional[str] = None
    modality: Optional[str] = None
    findings: List[Dict] = None
    confidence_scores: Dict[str, float] = None
    medical_summary: Optional[str] = None
    recommendations: List[str] = None
    dicom_metadata: Optional[Dict] = None
    normal_probability: Optional[str] = None
    processing_notes: List[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = []
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.recommendations is None:
            self.recommendations = []
        if self.processing_notes is None:
            self.processing_notes = []
        if self.created_at is None:
            self.created_at = datetime.now()

class MedicalDatabase:
    """SQLite database manager for medical test data"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Main analysis results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT,
                    file_type TEXT CHECK(file_type IN ('DICOM', 'Image')),
                    patient_id TEXT,
                    study_date TEXT,
                    modality TEXT,
                    findings TEXT,  -- JSON array
                    confidence_scores TEXT,  -- JSON object
                    medical_summary TEXT,
                    recommendations TEXT,  -- JSON array
                    dicom_metadata TEXT,  -- JSON object
                    normal_probability TEXT,
                    processing_notes TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Individual findings table for easier querying
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    condition_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity TEXT,
                    clinical_significance TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_results (id)
                )
            ''')
            
            # Statistics table for dashboard
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE,
                    total_analyses INTEGER DEFAULT 0,
                    dicom_analyses INTEGER DEFAULT 0,
                    image_analyses INTEGER DEFAULT 0,
                    abnormal_findings INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_date ON analysis_results(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results(file_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patient_id ON analysis_results(patient_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_findings_condition ON findings(condition_name)')
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    def save_analysis_result(self, result: AnalysisResult) -> int:
        """Save analysis result to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data for insertion
                cursor.execute('''
                    INSERT INTO analysis_results (
                        file_name, file_type, patient_id, study_date, modality,
                        findings, confidence_scores, medical_summary, recommendations,
                        dicom_metadata, normal_probability, processing_notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.file_name,
                    result.file_type,
                    result.patient_id,
                    result.study_date,
                    result.modality,
                    json.dumps(result.findings),
                    json.dumps(result.confidence_scores),
                    result.medical_summary,
                    json.dumps(result.recommendations),
                    json.dumps(result.dicom_metadata) if result.dicom_metadata else None,
                    result.normal_probability,
                    json.dumps(result.processing_notes),
                    result.created_at.isoformat()
                ))
                
                analysis_id = cursor.lastrowid
                
                # Save individual findings for easier querying
                for finding in result.findings:
                    cursor.execute('''
                        INSERT INTO findings (
                            analysis_id, condition_name, confidence, severity, clinical_significance
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (
                        analysis_id,
                        finding.get('condition'),
                        float(finding.get('confidence', '0%').rstrip('%')) / 100,
                        finding.get('severity'),
                        finding.get('clinical_significance')
                    ))
                
                # Update daily statistics
                self._update_daily_stats(result)
                
                conn.commit()
                logger.info(f"Analysis result saved with ID: {analysis_id}")
                return analysis_id
                
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
            raise
    
    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict]:
        """Retrieve analysis result by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM analysis_results WHERE id = ?', (analysis_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving analysis {analysis_id}: {e}")
            return None
    
    def get_recent_analyses(self, limit: int = 10) -> List[Dict]:
        """Get recent analysis results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                return [self._row_to_dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error retrieving recent analyses: {e}")
            return []
    
    def search_analyses(self, 
                       file_type: Optional[str] = None,
                       patient_id: Optional[str] = None,
                       date_from: Optional[str] = None,
                       date_to: Optional[str] = None,
                       has_findings: Optional[bool] = None) -> List[Dict]:
        """Search analysis results with filters"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM analysis_results WHERE 1=1"
                params = []
                
                if file_type:
                    query += " AND file_type = ?"
                    params.append(file_type)
                
                if patient_id:
                    query += " AND patient_id LIKE ?"
                    params.append(f"%{patient_id}%")
                
                if date_from:
                    query += " AND DATE(created_at) >= ?"
                    params.append(date_from)
                
                if date_to:
                    query += " AND DATE(created_at) <= ?"
                    params.append(date_to)
                
                if has_findings is not None:
                    if has_findings:
                        query += " AND findings != '[]'"
                    else:
                        query += " AND findings = '[]'"
                
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                return [self._row_to_dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error searching analyses: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total analyses
                cursor.execute('SELECT COUNT(*) as total FROM analysis_results')
                total = cursor.fetchone()['total']
                
                # By file type
                cursor.execute('SELECT file_type, COUNT(*) as count FROM analysis_results GROUP BY file_type')
                file_types = {row['file_type']: row['count'] for row in cursor.fetchall()}
                
                # Recent activity (last 7 days)
                cursor.execute('''
                    SELECT DATE(created_at) as date, COUNT(*) as count 
                    FROM analysis_results 
                    WHERE created_at >= datetime('now', '-7 days')
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                ''')
                recent_activity = [dict(row) for row in cursor.fetchall()]
                
                # Most common findings
                cursor.execute('''
                    SELECT condition_name, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM findings 
                    GROUP BY condition_name 
                    ORDER BY count DESC 
                    LIMIT 10
                ''')
                common_findings = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'total_analyses': total,
                    'file_types': file_types,
                    'recent_activity': recent_activity,
                    'common_findings': common_findings
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def delete_analysis(self, analysis_id: int) -> bool:
        """Delete analysis result"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete related findings first
                cursor.execute('DELETE FROM findings WHERE analysis_id = ?', (analysis_id,))
                
                # Delete analysis result
                cursor.execute('DELETE FROM analysis_results WHERE id = ?', (analysis_id,))
                
                conn.commit()
                logger.info(f"Analysis {analysis_id} deleted successfully")
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error deleting analysis {analysis_id}: {e}")
            return False
    
    def _row_to_dict(self, row) -> Dict:
        """Convert SQLite row to dictionary with JSON parsing"""
        result = dict(row)
        
        # Parse JSON fields
        json_fields = ['findings', 'confidence_scores', 'recommendations', 'dicom_metadata', 'processing_notes']
        for field in json_fields:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = None
        
        return result
    
    def _update_daily_stats(self, result: AnalysisResult):
        """Update daily statistics"""
        try:
            date_str = result.created_at.date().isoformat()
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert or update daily stats
                cursor.execute('''
                    INSERT OR IGNORE INTO analysis_stats (date, total_analyses, dicom_analyses, image_analyses, abnormal_findings)
                    VALUES (?, 0, 0, 0, 0)
                ''', (date_str,))
                
                # Update counters
                dicom_inc = 1 if result.file_type == 'DICOM' else 0
                image_inc = 1 if result.file_type == 'Image' else 0
                abnormal_inc = 1 if result.findings else 0
                
                cursor.execute('''
                    UPDATE analysis_stats 
                    SET total_analyses = total_analyses + 1,
                        dicom_analyses = dicom_analyses + ?,
                        image_analyses = image_analyses + ?,
                        abnormal_findings = abnormal_findings + ?
                    WHERE date = ?
                ''', (dicom_inc, image_inc, abnormal_inc, date_str))
                
        except Exception as e:
            logger.error(f"Error updating daily stats: {e}")

# Global database instance
db = MedicalDatabase()

def create_analysis_result_from_api_response(api_response: Dict, 
                                           file_name: str,
                                           file_type: str) -> AnalysisResult:
    """Helper function to create AnalysisResult from API response"""
    
    dicom_metadata = api_response.get('dicom_metadata', {})
    
    return AnalysisResult(
        file_name=file_name,
        file_type=file_type,
        patient_id=dicom_metadata.get('patient_id') if dicom_metadata else None,
        study_date=dicom_metadata.get('study_date') if dicom_metadata else None,
        modality=dicom_metadata.get('modality') if dicom_metadata else None,
        findings=api_response.get('findings', []),
        confidence_scores=api_response.get('confidence_scores', {}),
        medical_summary=api_response.get('medical_summary'),
        recommendations=api_response.get('recommendations', []),
        dicom_metadata=dicom_metadata,
        normal_probability=api_response.get('normal_probability'),
        processing_notes=api_response.get('processing_notes', [])
    )

if __name__ == "__main__":
    # Test database functionality
    db = MedicalDatabase()
    print("Database initialized successfully")
    
    # Print statistics
    stats = db.get_statistics()
    print(f"Database statistics: {stats}")
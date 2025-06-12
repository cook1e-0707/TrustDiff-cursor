"""
Data storage logic for TrustDiff using SQLite and JSON.
"""

import json
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import RawResult, EvaluationResult, TestSummary


class Storage:
    """Storage manager for TrustDiff results using SQLite + JSON files."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.output_dir / "results.db"
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        self._db_initialized = False
    
    async def initialize(self):
        """Initialize the database schema."""
        if self._db_initialized:
            return
        
        # Run in thread to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None, self._create_tables
        )
        self._db_initialized = True
    
    def _create_tables(self):
        """Create database tables."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Raw results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    probe_id TEXT NOT NULL,
                    platform_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    latency_ms REAL,
                    tokens_used INTEGER,
                    cost_estimate REAL,
                    error_message TEXT,
                    json_log_path TEXT
                )
            """)
            
            # Evaluations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    probe_id TEXT NOT NULL,
                    target_platform TEXT NOT NULL,
                    baseline_platform TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    latency_diff_ms REAL,
                    cost_diff REAL,
                    tokens_diff INTEGER,
                    quality_verdict TEXT,
                    quality_confidence REAL,
                    quality_reasoning TEXT,
                    quality_score_baseline REAL,
                    quality_score_target REAL,
                    evaluation_success BOOLEAN NOT NULL,
                    error_message TEXT,
                    -- New detailed scoring fields
                    comparative_reasoning TEXT,
                    target_correctness INTEGER,
                    target_reasoning_depth INTEGER,
                    target_instruction_adherence INTEGER,
                    target_clarity_conciseness INTEGER,
                    baseline_correctness INTEGER,
                    baseline_reasoning_depth INTEGER,
                    baseline_instruction_adherence INTEGER,
                    baseline_clarity_conciseness INTEGER
                )
            """)
            
            # Test runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    total_probes INTEGER,
                    total_platforms INTEGER,
                    total_evaluations INTEGER,
                    success_rate REAL
                )
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    async def save_raw_result(self, result: RawResult):
        """Save a raw API result."""
        # Save JSON log
        json_filename = f"{result.probe_id}_{result.platform_name}_{result.timestamp.strftime('%H%M%S')}.json"
        json_path = self.logs_dir / json_filename
        
        json_data = {
            "probe_id": result.probe_id,
            "platform_name": result.platform_name,
            "timestamp": result.timestamp.isoformat(),
            "success": result.success,
            "response_data": result.response_data,
            "error_message": result.error_message,
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "cost_estimate": result.cost_estimate
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save to database
        await asyncio.get_event_loop().run_in_executor(
            None, self._insert_raw_result, result, str(json_path)
        )
    
    def _insert_raw_result(self, result: RawResult, json_log_path: str):
        """Insert raw result into database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO raw_results (
                    probe_id, platform_name, timestamp, success,
                    latency_ms, tokens_used, cost_estimate, error_message, json_log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.probe_id,
                result.platform_name,
                result.timestamp.isoformat(),
                result.success,
                result.latency_ms,
                result.tokens_used,
                result.cost_estimate,
                result.error_message,
                json_log_path
            ))
            conn.commit()
        finally:
            conn.close()
    
    async def save_evaluation(self, evaluation: EvaluationResult):
        """Save an evaluation result."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._insert_evaluation, evaluation
        )
    
    def _insert_evaluation(self, evaluation: EvaluationResult):
        """Insert evaluation into database."""
        conn = sqlite3.connect(self.db_path)
        try:
            quality_verdict = None
            quality_confidence = None
            quality_reasoning = None
            quality_score_baseline = None
            quality_score_target = None
            comparative_reasoning = None
            
            # Detailed scores
            target_correctness = None
            target_reasoning_depth = None
            target_instruction_adherence = None
            target_clarity_conciseness = None
            baseline_correctness = None
            baseline_reasoning_depth = None
            baseline_instruction_adherence = None
            baseline_clarity_conciseness = None
            
            if evaluation.quality_evaluation:
                quality_verdict = evaluation.quality_evaluation.verdict
                quality_confidence = evaluation.quality_evaluation.confidence
                quality_reasoning = evaluation.quality_evaluation.reasoning
                quality_score_baseline = evaluation.quality_evaluation.score_baseline
                quality_score_target = evaluation.quality_evaluation.score_target
                comparative_reasoning = evaluation.quality_evaluation.comparative_reasoning
                
                # Extract detailed scores
                if evaluation.quality_evaluation.detailed_scores_target:
                    target_scores = evaluation.quality_evaluation.detailed_scores_target
                    target_correctness = target_scores.correctness
                    target_reasoning_depth = target_scores.reasoning_depth
                    target_instruction_adherence = target_scores.instruction_adherence
                    target_clarity_conciseness = target_scores.clarity_conciseness
                
                if evaluation.quality_evaluation.detailed_scores_baseline:
                    baseline_scores = evaluation.quality_evaluation.detailed_scores_baseline
                    baseline_correctness = baseline_scores.correctness
                    baseline_reasoning_depth = baseline_scores.reasoning_depth
                    baseline_instruction_adherence = baseline_scores.instruction_adherence
                    baseline_clarity_conciseness = baseline_scores.clarity_conciseness
            
            conn.execute("""
                INSERT INTO evaluations (
                    probe_id, target_platform, baseline_platform, timestamp,
                    latency_diff_ms, cost_diff, tokens_diff,
                    quality_verdict, quality_confidence, quality_reasoning,
                    quality_score_baseline, quality_score_target,
                    evaluation_success, error_message,
                    comparative_reasoning,
                    target_correctness, target_reasoning_depth, 
                    target_instruction_adherence, target_clarity_conciseness,
                    baseline_correctness, baseline_reasoning_depth,
                    baseline_instruction_adherence, baseline_clarity_conciseness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation.probe_id,
                evaluation.target_platform,
                evaluation.baseline_platform,
                evaluation.timestamp.isoformat(),
                evaluation.latency_diff_ms,
                evaluation.cost_diff,
                evaluation.tokens_diff,
                quality_verdict,
                quality_confidence,
                quality_reasoning,
                quality_score_baseline,
                quality_score_target,
                evaluation.evaluation_success,
                evaluation.error_message,
                comparative_reasoning,
                target_correctness,
                target_reasoning_depth,
                target_instruction_adherence,
                target_clarity_conciseness,
                baseline_correctness,
                baseline_reasoning_depth,
                baseline_instruction_adherence,
                baseline_clarity_conciseness
            ))
            conn.commit()
        finally:
            conn.close()
    
    async def save_test_summary(self, summary: TestSummary):
        """Save test run summary."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._insert_test_summary, summary
        )
    
    def _insert_test_summary(self, summary: TestSummary):
        """Insert test summary into database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO test_runs (
                    run_id, timestamp, total_probes, total_platforms,
                    total_evaluations, success_rate
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                summary.run_id,
                summary.timestamp.isoformat(),
                summary.total_probes,
                summary.total_platforms,
                summary.total_evaluations,
                summary.success_rate
            ))
            conn.commit()
        finally:
            conn.close()
    
    async def get_evaluations(self) -> List[Dict[str, Any]]:
        """Get all evaluations from database."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_evaluations
        )
    
    def _fetch_evaluations(self) -> List[Dict[str, Any]]:
        """Fetch evaluations from database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM evaluations ORDER BY timestamp")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    async def get_raw_results(self) -> List[Dict[str, Any]]:
        """Get all raw results from database."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_raw_results
        )
    
    def _fetch_raw_results(self) -> List[Dict[str, Any]]:
        """Fetch raw results from database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM raw_results ORDER BY timestamp")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Get basic counts
            cursor = conn.execute("SELECT COUNT(*) FROM evaluations")
            total_evaluations = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM evaluations WHERE evaluation_success = 1")
            successful_evaluations = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT probe_id) FROM evaluations")
            unique_probes = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT target_platform) FROM evaluations")
            unique_platforms = cursor.fetchone()[0]
            
            # Quality verdict distribution
            cursor = conn.execute("""
                SELECT quality_verdict, COUNT(*) 
                FROM evaluations 
                WHERE quality_verdict IS NOT NULL 
                GROUP BY quality_verdict
            """)
            quality_distribution = dict(cursor.fetchall())
            
            # Average metrics
            cursor = conn.execute("""
                SELECT 
                    AVG(latency_diff_ms) as avg_latency_diff,
                    AVG(cost_diff) as avg_cost_diff,
                    AVG(quality_confidence) as avg_quality_confidence
                FROM evaluations 
                WHERE evaluation_success = 1
            """)
            avg_metrics = cursor.fetchone()
            
            return {
                "total_evaluations": total_evaluations,
                "successful_evaluations": successful_evaluations,
                "success_rate": successful_evaluations / total_evaluations if total_evaluations > 0 else 0,
                "unique_probes": unique_probes,
                "unique_platforms": unique_platforms,
                "quality_distribution": quality_distribution,
                "avg_latency_diff_ms": avg_metrics[0],
                "avg_cost_diff": avg_metrics[1],
                "avg_quality_confidence": avg_metrics[2]
            }
        finally:
            conn.close() 
"""
Data storage logic for TrustDiff using SQLite and JSON.
Enhanced to support H-CAF (Hierarchical Cognitive Assessment Framework) data.
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
        """Create database tables with H-CAF support."""
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
            
            # Enhanced evaluations table with H-CAF fields
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
                    
                    -- Legacy detailed scoring fields (1-5 scale)
                    comparative_reasoning TEXT,
                    target_correctness INTEGER,
                    target_reasoning_depth INTEGER,
                    target_instruction_adherence INTEGER,
                    target_clarity_conciseness INTEGER,
                    baseline_correctness INTEGER,
                    baseline_reasoning_depth INTEGER,
                    baseline_instruction_adherence INTEGER,
                    baseline_clarity_conciseness INTEGER,
                    
                    -- H-CAF cognitive fingerprint fields (1-10 scale)
                    cognitive_focus TEXT,  -- JSON array of focus vectors
                    hcaf_target_logical_reasoning INTEGER,
                    hcaf_target_knowledge_application INTEGER,
                    hcaf_target_creative_synthesis INTEGER,
                    hcaf_target_instructional_fidelity INTEGER,
                    hcaf_target_safety_metacognition INTEGER,
                    hcaf_baseline_logical_reasoning INTEGER,
                    hcaf_baseline_knowledge_application INTEGER,
                    hcaf_baseline_creative_synthesis INTEGER,
                    hcaf_baseline_instructional_fidelity INTEGER,
                    hcaf_baseline_safety_metacognition INTEGER,
                    
                    -- H-CAF capability gaps
                    gap_logical_reasoning REAL,
                    gap_knowledge_application REAL,
                    gap_creative_synthesis REAL,
                    gap_instructional_fidelity REAL,
                    gap_safety_metacognition REAL,
                    
                    -- H-CAF metadata
                    comparative_audit_summary TEXT,
                    degradation_severity TEXT,
                    overall_degradation_score REAL
                )
            """)
            
            # Test runs table with H-CAF summary
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    total_probes INTEGER,
                    total_platforms INTEGER,
                    total_evaluations INTEGER,
                    success_rate REAL,
                    
                    -- H-CAF summary fields
                    total_hcaf_evaluations INTEGER,
                    avg_overall_degradation REAL,
                    most_degraded_vector TEXT,
                    cognitive_stability_rating TEXT
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
        """Save an evaluation result with H-CAF support."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._insert_evaluation, evaluation
        )
    
    def _insert_evaluation(self, evaluation: EvaluationResult):
        """Insert evaluation into database with H-CAF data."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Basic quality evaluation fields
            quality_verdict = None
            quality_confidence = None
            quality_reasoning = None
            quality_score_baseline = None
            quality_score_target = None
            comparative_reasoning = None
            
            # Legacy detailed scores (1-5 scale)
            target_correctness = None
            target_reasoning_depth = None
            target_instruction_adherence = None
            target_clarity_conciseness = None
            baseline_correctness = None
            baseline_reasoning_depth = None
            baseline_instruction_adherence = None
            baseline_clarity_conciseness = None
            
            # H-CAF fields
            cognitive_focus = None
            hcaf_target_logical_reasoning = None
            hcaf_target_knowledge_application = None
            hcaf_target_creative_synthesis = None
            hcaf_target_instructional_fidelity = None
            hcaf_target_safety_metacognition = None
            hcaf_baseline_logical_reasoning = None
            hcaf_baseline_knowledge_application = None
            hcaf_baseline_creative_synthesis = None
            hcaf_baseline_instructional_fidelity = None
            hcaf_baseline_safety_metacognition = None
            gap_logical_reasoning = None
            gap_knowledge_application = None
            gap_creative_synthesis = None
            gap_instructional_fidelity = None
            gap_safety_metacognition = None
            comparative_audit_summary = None
            degradation_severity = None
            overall_degradation_score = None
            
            if evaluation.quality_evaluation:
                quality_verdict = evaluation.quality_evaluation.verdict
                quality_confidence = evaluation.quality_evaluation.confidence
                quality_reasoning = evaluation.quality_evaluation.reasoning
                quality_score_baseline = evaluation.quality_evaluation.score_baseline
                quality_score_target = evaluation.quality_evaluation.score_target
                comparative_reasoning = evaluation.quality_evaluation.comparative_reasoning
                
                # H-CAF cognitive focus
                if evaluation.quality_evaluation.cognitive_focus:
                    cognitive_focus = json.dumps(evaluation.quality_evaluation.cognitive_focus)
                
                # H-CAF comparative audit summary
                comparative_audit_summary = evaluation.quality_evaluation.comparative_audit_summary
                
                # Extract H-CAF cognitive fingerprints
                if evaluation.quality_evaluation.cognitive_fingerprint_target:
                    target_cf = evaluation.quality_evaluation.cognitive_fingerprint_target
                    hcaf_target_logical_reasoning = target_cf.logical_reasoning
                    hcaf_target_knowledge_application = target_cf.knowledge_application
                    hcaf_target_creative_synthesis = target_cf.creative_synthesis
                    hcaf_target_instructional_fidelity = target_cf.instructional_fidelity
                    hcaf_target_safety_metacognition = target_cf.safety_metacognition
                
                if evaluation.quality_evaluation.cognitive_fingerprint_baseline:
                    baseline_cf = evaluation.quality_evaluation.cognitive_fingerprint_baseline
                    hcaf_baseline_logical_reasoning = baseline_cf.logical_reasoning
                    hcaf_baseline_knowledge_application = baseline_cf.knowledge_application
                    hcaf_baseline_creative_synthesis = baseline_cf.creative_synthesis
                    hcaf_baseline_instructional_fidelity = baseline_cf.instructional_fidelity
                    hcaf_baseline_safety_metacognition = baseline_cf.safety_metacognition
                
                # Extract H-CAF capability gaps
                if evaluation.quality_evaluation.capability_gaps:
                    gaps = evaluation.quality_evaluation.capability_gaps
                    gap_logical_reasoning = gaps.logical_reasoning_gap
                    gap_knowledge_application = gaps.knowledge_application_gap
                    gap_creative_synthesis = gaps.creative_synthesis_gap
                    gap_instructional_fidelity = gaps.instructional_fidelity_gap
                    gap_safety_metacognition = gaps.safety_metacognition_gap
                    overall_degradation_score = gaps.get_overall_degradation()
                
                # Extract degradation severity
                degradation_severity = evaluation.quality_evaluation.get_degradation_severity()
                
                # Extract legacy detailed scores
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
                    baseline_instruction_adherence, baseline_clarity_conciseness,
                    cognitive_focus,
                    hcaf_target_logical_reasoning, hcaf_target_knowledge_application,
                    hcaf_target_creative_synthesis, hcaf_target_instructional_fidelity,
                    hcaf_target_safety_metacognition,
                    hcaf_baseline_logical_reasoning, hcaf_baseline_knowledge_application,
                    hcaf_baseline_creative_synthesis, hcaf_baseline_instructional_fidelity,
                    hcaf_baseline_safety_metacognition,
                    gap_logical_reasoning, gap_knowledge_application,
                    gap_creative_synthesis, gap_instructional_fidelity,
                    gap_safety_metacognition,
                    comparative_audit_summary, degradation_severity, overall_degradation_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                baseline_clarity_conciseness,
                cognitive_focus,
                hcaf_target_logical_reasoning,
                hcaf_target_knowledge_application,
                hcaf_target_creative_synthesis,
                hcaf_target_instructional_fidelity,
                hcaf_target_safety_metacognition,
                hcaf_baseline_logical_reasoning,
                hcaf_baseline_knowledge_application,
                hcaf_baseline_creative_synthesis,
                hcaf_baseline_instructional_fidelity,
                hcaf_baseline_safety_metacognition,
                gap_logical_reasoning,
                gap_knowledge_application,
                gap_creative_synthesis,
                gap_instructional_fidelity,
                gap_safety_metacognition,
                comparative_audit_summary,
                degradation_severity,
                overall_degradation_score
            ))
            conn.commit()
        finally:
            conn.close()
    
    async def save_test_summary(self, summary: TestSummary):
        """Save test run summary with H-CAF data."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._insert_test_summary, summary
        )
    
    def _insert_test_summary(self, summary: TestSummary):
        """Insert test summary into database with H-CAF data."""
        conn = sqlite3.connect(self.db_path)
        try:
            # H-CAF summary fields
            total_hcaf_evaluations = None
            avg_overall_degradation = None
            most_degraded_vector = None
            cognitive_stability_rating = None
            
            if summary.cognitive_benchmark_summary:
                hcaf_summary = summary.cognitive_benchmark_summary
                total_hcaf_evaluations = hcaf_summary.total_cognitive_evaluations
                avg_overall_degradation = hcaf_summary.overall_degradation_score
                most_degraded_vector = hcaf_summary.most_degraded_vectors[0] if hcaf_summary.most_degraded_vectors else None
                cognitive_stability_rating = hcaf_summary.cognitive_stability_rating
            
            conn.execute("""
                INSERT OR REPLACE INTO test_runs (
                    run_id, timestamp, total_probes, total_platforms,
                    total_evaluations, success_rate,
                    total_hcaf_evaluations, avg_overall_degradation,
                    most_degraded_vector, cognitive_stability_rating
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.run_id,
                summary.timestamp.isoformat(),
                summary.total_probes,
                summary.total_platforms,
                summary.total_evaluations,
                summary.success_rate,
                total_hcaf_evaluations,
                avg_overall_degradation,
                most_degraded_vector,
                cognitive_stability_rating
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
        """Get summary statistics with H-CAF metrics."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Basic counts
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
            
            # Basic average metrics
            cursor = conn.execute("""
                SELECT 
                    AVG(latency_diff_ms) as avg_latency_diff,
                    AVG(cost_diff) as avg_cost_diff,
                    AVG(quality_confidence) as avg_quality_confidence
                FROM evaluations 
                WHERE evaluation_success = 1
            """)
            avg_metrics = cursor.fetchone()
            
            # H-CAF specific metrics
            cursor = conn.execute("""
                SELECT COUNT(*) FROM evaluations 
                WHERE hcaf_target_logical_reasoning IS NOT NULL
            """)
            total_hcaf_evaluations = cursor.fetchone()[0]
            
            # H-CAF cognitive vector averages
            hcaf_averages = {}
            if total_hcaf_evaluations > 0:
                cursor = conn.execute("""
                    SELECT 
                        AVG(hcaf_target_logical_reasoning) as avg_target_logical,
                        AVG(hcaf_baseline_logical_reasoning) as avg_baseline_logical,
                        AVG(hcaf_target_knowledge_application) as avg_target_knowledge,
                        AVG(hcaf_baseline_knowledge_application) as avg_baseline_knowledge,
                        AVG(hcaf_target_creative_synthesis) as avg_target_creative,
                        AVG(hcaf_baseline_creative_synthesis) as avg_baseline_creative,
                        AVG(hcaf_target_instructional_fidelity) as avg_target_instruction,
                        AVG(hcaf_baseline_instructional_fidelity) as avg_baseline_instruction,
                        AVG(hcaf_target_safety_metacognition) as avg_target_safety,
                        AVG(hcaf_baseline_safety_metacognition) as avg_baseline_safety,
                        AVG(overall_degradation_score) as avg_overall_degradation
                    FROM evaluations 
                    WHERE hcaf_target_logical_reasoning IS NOT NULL
                """)
                hcaf_data = cursor.fetchone()
                
                hcaf_averages = {
                    "avg_target_logical_reasoning": hcaf_data[0],
                    "avg_baseline_logical_reasoning": hcaf_data[1],
                    "avg_target_knowledge_application": hcaf_data[2],
                    "avg_baseline_knowledge_application": hcaf_data[3],
                    "avg_target_creative_synthesis": hcaf_data[4],
                    "avg_baseline_creative_synthesis": hcaf_data[5],
                    "avg_target_instructional_fidelity": hcaf_data[6],
                    "avg_baseline_instructional_fidelity": hcaf_data[7],
                    "avg_target_safety_metacognition": hcaf_data[8],
                    "avg_baseline_safety_metacognition": hcaf_data[9],
                    "avg_overall_degradation": hcaf_data[10]
                }
            
            # Degradation severity distribution
            cursor = conn.execute("""
                SELECT degradation_severity, COUNT(*) 
                FROM evaluations 
                WHERE degradation_severity IS NOT NULL 
                GROUP BY degradation_severity
            """)
            degradation_distribution = dict(cursor.fetchall())
            
            return {
                "total_evaluations": total_evaluations,
                "successful_evaluations": successful_evaluations,
                "success_rate": successful_evaluations / total_evaluations if total_evaluations > 0 else 0,
                "unique_probes": unique_probes,
                "unique_platforms": unique_platforms,
                "quality_distribution": quality_distribution,
                "avg_latency_diff_ms": avg_metrics[0],
                "avg_cost_diff": avg_metrics[1],
                "avg_quality_confidence": avg_metrics[2],
                # H-CAF metrics
                "total_hcaf_evaluations": total_hcaf_evaluations,
                "hcaf_averages": hcaf_averages,
                "degradation_distribution": degradation_distribution
            }
        finally:
            conn.close() 
"""
Enhanced storage module for TrustDiff H-CAF framework.
Supports both traditional evaluation results and H-CAF cognitive assessments.
"""

import os
import json
import yaml
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .models import (
    TrustDiffReport, EvaluationResult, QualityEvaluation, 
    RawResult, ExecutionPlan, CognitiveFingerprint, CapabilityGaps
)


class TrustDiffStorage:
    """Enhanced storage manager supporting H-CAF cognitive assessment data"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.ensure_output_directory()
        
        self.db_path = self.output_dir / "results.db"
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        self._db_initialized = False
    
    def ensure_output_directory(self):
        """Ensure output directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "raw_responses"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "hcaf_reports"), exist_ok=True)
    
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
    
    def save_report(self, report: TrustDiffReport, filename_prefix: str = "trustdiff_report") -> str:
        """Save complete TrustDiff report with H-CAF support"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        # Convert to serializable format
        report_dict = self._prepare_report_for_serialization(report)
        
        # Save as JSON
        json_file = os.path.join(self.output_dir, f"{base_filename}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # Save as YAML for better readability
        yaml_file = os.path.join(self.output_dir, f"{base_filename}.yaml")
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(report_dict, f, default_flow_style=False, allow_unicode=True)
        
        # Save H-CAF specific analysis
        if self._has_hcaf_data(report):
            hcaf_file = os.path.join(self.output_dir, "hcaf_reports", f"hcaf_{base_filename}.json")
            hcaf_analysis = self._extract_hcaf_analysis(report)
            with open(hcaf_file, 'w', encoding='utf-8') as f:
                json.dump(hcaf_analysis, f, indent=2, ensure_ascii=False, default=str)
        
        return json_file
    
    def _prepare_report_for_serialization(self, report: TrustDiffReport) -> Dict[str, Any]:
        """Convert TrustDiffReport to JSON-serializable format"""
        return {
            'metadata': {
                'framework_version': 'H-CAF v1.0',
                'execution_timestamp': report.execution_timestamp.isoformat(),
                'total_runtime_seconds': report.total_runtime_seconds,
                'target_platform': report.execution_plan.target_platform.name,
                'baseline_platform': report.execution_plan.baseline_platform.name
            },
            'execution_plan': asdict(report.execution_plan),
            'summary_statistics': {
                'total_probes': len(report.evaluation_results),
                'success_rate_target': report.success_rate_target,
                'success_rate_baseline': report.success_rate_baseline,
                'evaluation_success_rate': report.evaluation_success_rate,
                'cognitive_performance': report.get_cognitive_performance_summary()
            },
            'evaluation_results': [self._serialize_evaluation_result(result) for result in report.evaluation_results],
            'raw_results': {
                'target': [asdict(result) for result in report.raw_results_target],
                'baseline': [asdict(result) for result in report.raw_results_baseline]
            },
            'executive_summary': report.get_executive_summary()
        }
    
    def _serialize_evaluation_result(self, result: EvaluationResult) -> Dict[str, Any]:
        """Serialize single evaluation result with H-CAF support"""
        result_dict = {
            'probe_id': result.probe_id,
            'target_platform': result.target_platform,
            'baseline_platform': result.baseline_platform,
            'evaluation_success': result.evaluation_success,
            'performance_metrics': {
                'latency_diff_ms': result.latency_diff_ms,
                'cost_diff': result.cost_diff,
                'tokens_diff': result.tokens_diff
            }
        }
        
        if result.quality_evaluation:
            quality_dict = {
                'verdict': result.quality_evaluation.verdict,
                'confidence': result.quality_evaluation.confidence,
                'reasoning': result.quality_evaluation.reasoning,
                'evaluation_type': 'H-CAF' if result.quality_evaluation.is_hcaf_evaluation() else 'Legacy'
            }
            
            # Add H-CAF specific data if available
            if result.quality_evaluation.is_hcaf_evaluation():
                quality_dict.update({
                    'cognitive_focus': result.quality_evaluation.cognitive_focus,
                    'cognitive_fingerprint_target': asdict(result.quality_evaluation.cognitive_fingerprint_target),
                    'cognitive_fingerprint_baseline': asdict(result.quality_evaluation.cognitive_fingerprint_baseline),
                    'capability_gaps': asdict(result.quality_evaluation.capability_gaps),
                    'comparative_audit_summary': result.quality_evaluation.comparative_audit_summary,
                    'degradation_severity': result.quality_evaluation.get_degradation_severity()
                })
            
            # Add legacy data if available
            elif result.quality_evaluation.detailed_scores_target:
                quality_dict.update({
                    'detailed_scores_target': asdict(result.quality_evaluation.detailed_scores_target),
                    'detailed_scores_baseline': asdict(result.quality_evaluation.detailed_scores_baseline),
                    'score_target': result.quality_evaluation.score_target,
                    'score_baseline': result.quality_evaluation.score_baseline,
                    'comparative_reasoning': result.quality_evaluation.comparative_reasoning
                })
            
            result_dict['quality_evaluation'] = quality_dict
        
        if result.error_message:
            result_dict['error_message'] = result.error_message
        
        return result_dict
    
    def _has_hcaf_data(self, report: TrustDiffReport) -> bool:
        """Check if report contains H-CAF evaluation data"""
        return any(
            result.quality_evaluation and result.quality_evaluation.is_hcaf_evaluation()
            for result in report.evaluation_results
        )
    
    def _extract_hcaf_analysis(self, report: TrustDiffReport) -> Dict[str, Any]:
        """Extract H-CAF specific analysis from report"""
        hcaf_results = [
            result for result in report.evaluation_results
            if result.quality_evaluation and result.quality_evaluation.is_hcaf_evaluation()
        ]
        
        if not hcaf_results:
            return {"note": "No H-CAF evaluation data available"}
        
        # Detailed cognitive analysis
        cognitive_dimensions = ['logical_reasoning', 'knowledge_application', 'creative_synthesis', 
                              'instructional_fidelity', 'safety_metacognition']
        
        dimension_analysis = {}
        for dimension in cognitive_dimensions:
            target_scores = []
            baseline_scores = []
            gaps = []
            
            for result in hcaf_results:
                if result.quality_evaluation.cognitive_fingerprint_target:
                    target_scores.append(getattr(result.quality_evaluation.cognitive_fingerprint_target, dimension))
                if result.quality_evaluation.cognitive_fingerprint_baseline:
                    baseline_scores.append(getattr(result.quality_evaluation.cognitive_fingerprint_baseline, dimension))
                if result.quality_evaluation.capability_gaps:
                    gaps.append(getattr(result.quality_evaluation.capability_gaps, f"{dimension}_gap"))
            
            dimension_analysis[dimension] = {
                'target_average': sum(target_scores) / len(target_scores) if target_scores else 0,
                'baseline_average': sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
                'average_gap': sum(gaps) / len(gaps) if gaps else 0,
                'sample_count': len(target_scores)
            }
        
        # Probe-level analysis
        probe_analysis = []
        for result in hcaf_results:
            probe_data = {
                'probe_id': result.probe_id,
                'verdict': result.quality_evaluation.verdict,
                'degradation_severity': result.quality_evaluation.get_degradation_severity(),
                'confidence': result.quality_evaluation.confidence
            }
            
            if result.quality_evaluation.cognitive_fingerprint_target:
                probe_data['target_cognitive_score'] = result.quality_evaluation.cognitive_fingerprint_target.get_average_score()
            
            if result.quality_evaluation.cognitive_fingerprint_baseline:
                probe_data['baseline_cognitive_score'] = result.quality_evaluation.cognitive_fingerprint_baseline.get_average_score()
            
            if result.quality_evaluation.capability_gaps:
                probe_data['overall_degradation'] = result.quality_evaluation.capability_gaps.get_average_degradation()
                probe_data['major_weaknesses'] = result.quality_evaluation.capability_gaps.get_major_weaknesses()
            
            probe_analysis.append(probe_data)
        
        return {
            'hcaf_framework_version': 'v1.0',
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_hcaf_evaluations': len(hcaf_results),
                'overall_performance': report.get_cognitive_performance_summary()
            },
            'cognitive_dimension_analysis': dimension_analysis,
            'probe_level_analysis': probe_analysis,
            'recommendations': self._generate_hcaf_recommendations(dimension_analysis)
        }
    
    def _generate_hcaf_recommendations(self, dimension_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on H-CAF analysis"""
        recommendations = []
        
        # Find the most problematic dimensions
        sorted_dimensions = sorted(
            dimension_analysis.items(),
            key=lambda x: x[1]['average_gap'],
            reverse=True
        )
        
        for dimension, analysis in sorted_dimensions[:3]:  # Top 3 problematic areas
            gap = analysis['average_gap']
            if gap > 1.0:
                recommendations.append(
                    f"Critical improvement needed in {dimension.replace('_', ' ')}: "
                    f"average degradation of {gap:.2f} points"
                )
            elif gap > 0.5:
                recommendations.append(
                    f"Monitor {dimension.replace('_', ' ')}: "
                    f"moderate degradation of {gap:.2f} points"
                )
        
        # Overall performance recommendations
        overall_gap = sum(d['average_gap'] for d in dimension_analysis.values()) / len(dimension_analysis)
        if overall_gap > 1.5:
            recommendations.append("Consider comprehensive model retraining or architecture revision")
        elif overall_gap > 0.8:
            recommendations.append("Implement targeted fine-tuning for identified weak areas")
        elif overall_gap < -0.5:
            recommendations.append("Performance improvement detected - consider broader deployment")
        
        return recommendations if recommendations else ["No significant issues detected"]
    
    def save_raw_responses(self, raw_results: List[RawResult]) -> List[str]:
        """Save raw API responses for detailed analysis"""
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for result in raw_results:
            if result.response_data:
                filename = f"raw_{result.platform_name}_{result.probe_id}_{timestamp}.json"
                filepath = os.path.join(self.output_dir, "raw_responses", filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        'probe_id': result.probe_id,
                        'platform_name': result.platform_name,
                        'success': result.success,
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                        'latency_ms': result.latency_ms,
                        'tokens_used': result.tokens_used,
                        'cost_estimate': result.cost_estimate,
                        'response_data': result.response_data,
                        'error_message': result.error_message
                    }, f, indent=2, ensure_ascii=False, default=str)
                
                saved_files.append(filepath)
        
        return saved_files
    
    def create_database(self, db_path: str = None) -> str:
        """Create SQLite database with H-CAF support for advanced analysis"""
        if db_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_path = os.path.join(self.output_dir, f"trustdiff_hcaf_{timestamp}.db")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Main evaluation results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                probe_id TEXT NOT NULL,
                target_platform TEXT NOT NULL,
                baseline_platform TEXT NOT NULL,
                evaluation_success BOOLEAN NOT NULL,
                latency_diff_ms REAL,
                cost_diff REAL,
                tokens_diff INTEGER,
                verdict TEXT,
                confidence REAL,
                reasoning TEXT,
                evaluation_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # H-CAF cognitive fingerprints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cognitive_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                platform_type TEXT NOT NULL, -- 'target' or 'baseline'
                logical_reasoning REAL NOT NULL,
                knowledge_application REAL NOT NULL,
                creative_synthesis REAL NOT NULL,
                instructional_fidelity REAL NOT NULL,
                safety_metacognition REAL NOT NULL,
                total_score REAL NOT NULL,
                average_score REAL NOT NULL,
                FOREIGN KEY (evaluation_id) REFERENCES evaluation_results (id)
            )
        ''')
        
        # H-CAF capability gaps table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capability_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                logical_reasoning_gap REAL NOT NULL,
                knowledge_application_gap REAL NOT NULL,
                creative_synthesis_gap REAL NOT NULL,
                instructional_fidelity_gap REAL NOT NULL,
                safety_metacognition_gap REAL NOT NULL,
                total_degradation REAL NOT NULL,
                average_degradation REAL NOT NULL,
                degradation_severity TEXT,
                FOREIGN KEY (evaluation_id) REFERENCES evaluation_results (id)
            )
        ''')
        
        # Raw results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                probe_id TEXT NOT NULL,
                platform_name TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                latency_ms REAL,
                tokens_used INTEGER,
                cost_estimate REAL,
                error_message TEXT,
                response_data_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        return db_path
    
    def save_to_database(self, report: TrustDiffReport, db_path: str = None) -> str:
        """Save TrustDiffReport to SQLite database with full H-CAF support"""
        if db_path is None:
            db_path = self.create_database()
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Save raw results
        for result in report.raw_results_target + report.raw_results_baseline:
            cursor.execute('''
                INSERT INTO raw_results 
                (probe_id, platform_name, success, latency_ms, tokens_used, cost_estimate, error_message, response_data_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.probe_id,
                result.platform_name,
                result.success,
                result.latency_ms,
                result.tokens_used,
                result.cost_estimate,
                result.error_message,
                json.dumps(result.response_data) if result.response_data else None,
                result.timestamp
            ))
        
        # Save evaluation results
        for eval_result in report.evaluation_results:
            # Insert main evaluation record
            cursor.execute('''
                INSERT INTO evaluation_results 
                (probe_id, target_platform, baseline_platform, evaluation_success, latency_diff_ms, cost_diff, tokens_diff, verdict, confidence, reasoning, evaluation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                eval_result.probe_id,
                eval_result.target_platform,
                eval_result.baseline_platform,
                eval_result.evaluation_success,
                eval_result.latency_diff_ms,
                eval_result.cost_diff,
                eval_result.tokens_diff,
                eval_result.quality_evaluation.verdict if eval_result.quality_evaluation else None,
                eval_result.quality_evaluation.confidence if eval_result.quality_evaluation else None,
                eval_result.quality_evaluation.reasoning if eval_result.quality_evaluation else None,
                'H-CAF' if eval_result.quality_evaluation and eval_result.quality_evaluation.is_hcaf_evaluation() else 'Legacy'
            ))
            
            evaluation_id = cursor.lastrowid
            
            # Save H-CAF data if available
            if eval_result.quality_evaluation and eval_result.quality_evaluation.is_hcaf_evaluation():
                # Save cognitive fingerprints
                if eval_result.quality_evaluation.cognitive_fingerprint_target:
                    fp = eval_result.quality_evaluation.cognitive_fingerprint_target
                    cursor.execute('''
                        INSERT INTO cognitive_fingerprints 
                        (evaluation_id, platform_type, logical_reasoning, knowledge_application, creative_synthesis, instructional_fidelity, safety_metacognition, total_score, average_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        evaluation_id, 'target', fp.logical_reasoning, fp.knowledge_application,
                        fp.creative_synthesis, fp.instructional_fidelity, fp.safety_metacognition,
                        fp.get_total_score(), fp.get_average_score()
                    ))
                
                if eval_result.quality_evaluation.cognitive_fingerprint_baseline:
                    fp = eval_result.quality_evaluation.cognitive_fingerprint_baseline
                    cursor.execute('''
                        INSERT INTO cognitive_fingerprints 
                        (evaluation_id, platform_type, logical_reasoning, knowledge_application, creative_synthesis, instructional_fidelity, safety_metacognition, total_score, average_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        evaluation_id, 'baseline', fp.logical_reasoning, fp.knowledge_application,
                        fp.creative_synthesis, fp.instructional_fidelity, fp.safety_metacognition,
                        fp.get_total_score(), fp.get_average_score()
                    ))
                
                # Save capability gaps
                if eval_result.quality_evaluation.capability_gaps:
                    gaps = eval_result.quality_evaluation.capability_gaps
                    cursor.execute('''
                        INSERT INTO capability_gaps 
                        (evaluation_id, logical_reasoning_gap, knowledge_application_gap, creative_synthesis_gap, instructional_fidelity_gap, safety_metacognition_gap, total_degradation, average_degradation, degradation_severity)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        evaluation_id, gaps.logical_reasoning_gap, gaps.knowledge_application_gap,
                        gaps.creative_synthesis_gap, gaps.instructional_fidelity_gap, gaps.safety_metacognition_gap,
                        gaps.get_total_degradation(), gaps.get_average_degradation(),
                        eval_result.quality_evaluation.get_degradation_severity()
                    ))
        
        conn.commit()
        conn.close()
        
        return db_path
    
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
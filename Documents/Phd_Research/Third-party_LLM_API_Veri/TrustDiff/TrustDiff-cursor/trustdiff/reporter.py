"""
Report generation logic for TrustDiff using Pandas.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from rich.console import Console

console = Console()


class Reporter:
    """Generate reports from TrustDiff test results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.db_path = self.results_dir / "results.db"
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Results database not found: {self.db_path}")
    
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from SQLite database."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            
            # Load evaluations
            evaluations_df = pd.read_sql_query("""
                SELECT * FROM evaluations 
                ORDER BY timestamp
            """, conn)
            
            # Load raw results
            raw_results_df = pd.read_sql_query("""
                SELECT * FROM raw_results 
                ORDER BY timestamp
            """, conn)
            
            conn.close()
            return evaluations_df, raw_results_df
            
        except Exception as e:
            console.print(f"[red]Failed to load data: {e}[/red]")
            return pd.DataFrame(), pd.DataFrame()
    
    def _generate_summary_stats(self, evaluations_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        if evaluations_df.empty:
            return {}
        
        stats = {
            "total_evaluations": len(evaluations_df),
            "successful_evaluations": len(evaluations_df[evaluations_df['evaluation_success'] == 1]),
            "unique_probes": evaluations_df['probe_id'].nunique(),
            "unique_platforms": evaluations_df['target_platform'].nunique(),
        }
        
        stats["success_rate"] = stats["successful_evaluations"] / stats["total_evaluations"]
        
        # Quality distribution
        if 'quality_verdict' in evaluations_df.columns:
            quality_counts = evaluations_df['quality_verdict'].value_counts()
            stats["quality_distribution"] = quality_counts.to_dict()
        
        # Performance metrics
        successful_evals = evaluations_df[evaluations_df['evaluation_success'] == 1]
        if not successful_evals.empty:
            stats["avg_latency_diff_ms"] = successful_evals['latency_diff_ms'].mean()
            stats["avg_cost_diff"] = successful_evals['cost_diff'].mean()
            stats["avg_quality_confidence"] = successful_evals['quality_confidence'].mean()
            
            # Detailed scoring analysis
            detailed_score_columns = [
                'target_correctness', 'target_reasoning_depth', 
                'target_instruction_adherence', 'target_clarity_conciseness',
                'baseline_correctness', 'baseline_reasoning_depth',
                'baseline_instruction_adherence', 'baseline_clarity_conciseness'
            ]
            
            available_columns = [col for col in detailed_score_columns if col in successful_evals.columns]
            if available_columns:
                detailed_stats = {}
                
                # Calculate score differences
                score_dimensions = ['correctness', 'reasoning_depth', 'instruction_adherence', 'clarity_conciseness']
                for dimension in score_dimensions:
                    target_col = f'target_{dimension}'
                    baseline_col = f'baseline_{dimension}'
                    
                    if target_col in successful_evals.columns and baseline_col in successful_evals.columns:
                        target_avg = successful_evals[target_col].mean()
                        baseline_avg = successful_evals[baseline_col].mean()
                        
                        detailed_stats[f'avg_{dimension}_target'] = target_avg
                        detailed_stats[f'avg_{dimension}_baseline'] = baseline_avg
                        detailed_stats[f'avg_{dimension}_diff'] = target_avg - baseline_avg
                        detailed_stats[f'avg_{dimension}_degradation_pct'] = ((baseline_avg - target_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
                
                stats["detailed_scoring"] = detailed_stats
        
        return stats
    
    def _generate_platform_comparison(self, evaluations_df: pd.DataFrame) -> pd.DataFrame:
        """Generate platform comparison table."""
        if evaluations_df.empty:
            return pd.DataFrame()
        
        # Group by platform
        platform_stats = evaluations_df.groupby('target_platform').agg({
            'evaluation_success': ['count', 'sum'],
            'latency_diff_ms': 'mean',
            'cost_diff': 'mean',
            'quality_confidence': 'mean',
            'quality_verdict': lambda x: (x == 'target_better').sum() if x.notna().any() else 0
        }).round(2)
        
        # Flatten column names
        platform_stats.columns = ['total_tests', 'successful_tests', 'avg_latency_diff', 
                                 'avg_cost_diff', 'avg_confidence', 'quality_wins']
        
        # Calculate success rate and win rate
        platform_stats['success_rate'] = (platform_stats['successful_tests'] / 
                                        platform_stats['total_tests']).round(3)
        platform_stats['win_rate'] = (platform_stats['quality_wins'] / 
                                     platform_stats['successful_tests']).round(3)
        
        return platform_stats.reset_index()
    
    def _generate_probe_analysis(self, evaluations_df: pd.DataFrame) -> pd.DataFrame:
        """Generate probe-level analysis."""
        if evaluations_df.empty:
            return pd.DataFrame()
        
        probe_stats = evaluations_df.groupby('probe_id').agg({
            'evaluation_success': ['count', 'sum'],
            'latency_diff_ms': 'mean',
            'quality_verdict': lambda x: (x == 'target_better').sum() if x.notna().any() else 0
        }).round(2)
        
        probe_stats.columns = ['total_platforms', 'successful_evals', 'avg_latency_diff', 'target_wins']
        probe_stats['success_rate'] = (probe_stats['successful_evals'] / 
                                     probe_stats['total_platforms']).round(3)
        
        return probe_stats.reset_index()
    
    def generate_markdown_report(self) -> str:
        """Generate a comprehensive Markdown report."""
        evaluations_df, raw_results_df = self._load_data()
        
        if evaluations_df.empty:
            return "# TrustDiff Report\n\nNo evaluation data found."
        
        # Generate components
        summary_stats = self._generate_summary_stats(evaluations_df)
        platform_comparison = self._generate_platform_comparison(evaluations_df)
        probe_analysis = self._generate_probe_analysis(evaluations_df)
        
        # Build report
        report = []
        report.append("# TrustDiff Test Report")
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nResults directory: `{self.results_dir}`")
        
        # Executive Summary
        report.append("\n## Executive Summary")
        report.append(f"- **Total Evaluations**: {summary_stats.get('total_evaluations', 0)}")
        report.append(f"- **Success Rate**: {summary_stats.get('success_rate', 0):.1%}")
        report.append(f"- **Unique Probes**: {summary_stats.get('unique_probes', 0)}")
        report.append(f"- **Platforms Tested**: {summary_stats.get('unique_platforms', 0)}")
        
        if 'avg_latency_diff_ms' in summary_stats:
            avg_latency = summary_stats['avg_latency_diff_ms']
            if avg_latency is not None:
                report.append(f"- **Average Latency Difference**: {avg_latency:.1f}ms")
        
        # Quality Distribution
        if 'quality_distribution' in summary_stats and summary_stats['quality_distribution']:
            report.append("\n### Quality Verdict Distribution")
            for verdict, count in summary_stats['quality_distribution'].items():
                if verdict:  # Skip None values
                    percentage = (count / summary_stats['total_evaluations']) * 100
                    report.append(f"- **{verdict.replace('_', ' ').title()}**: {count} ({percentage:.1f}%)")
        
        # Detailed Scoring Analysis
        if 'detailed_scoring' in summary_stats and summary_stats['detailed_scoring']:
            report.append("\n### Detailed Quality Analysis")
            detailed = summary_stats['detailed_scoring']
            
            report.append("Average scores comparison (1-5 scale):")
            report.append("")
            report.append("| Dimension | Target | Baseline | Difference | Degradation % |")
            report.append("|-----------|--------|----------|------------|---------------|")
            
            dimensions = ['correctness', 'reasoning_depth', 'instruction_adherence', 'clarity_conciseness']
            for dim in dimensions:
                target_key = f'avg_{dim}_target'
                baseline_key = f'avg_{dim}_baseline'
                diff_key = f'avg_{dim}_diff'
                degradation_key = f'avg_{dim}_degradation_pct'
                
                if all(key in detailed for key in [target_key, baseline_key, diff_key, degradation_key]):
                    target = detailed[target_key]
                    baseline = detailed[baseline_key]
                    diff = detailed[diff_key]
                    degradation = detailed[degradation_key]
                    
                    dim_name = dim.replace('_', ' ').title()
                    report.append(f"| {dim_name} | {target:.2f} | {baseline:.2f} | {diff:+.2f} | {degradation:.1f}% |")
            
            # Identify most problematic dimensions
            report.append("\n#### Key Findings")
            worst_dimension = None
            worst_degradation = 0
            
            for dim in dimensions:
                degradation_key = f'avg_{dim}_degradation_pct'
                if degradation_key in detailed and detailed[degradation_key] > worst_degradation:
                    worst_degradation = detailed[degradation_key]
                    worst_dimension = dim
            
            if worst_dimension and worst_degradation > 5:  # Only report if degradation > 5%
                dim_name = worst_dimension.replace('_', ' ').title()
                report.append(f"- **Most Affected Dimension**: {dim_name} shows {worst_degradation:.1f}% average degradation")
            
            # Overall quality trend
            total_degradations = [detailed.get(f'avg_{dim}_degradation_pct', 0) for dim in dimensions]
            avg_degradation = sum(total_degradations) / len(total_degradations)
            if avg_degradation > 0:
                report.append(f"- **Overall Quality Impact**: Average {avg_degradation:.1f}% degradation across all dimensions")
        
        # Platform Comparison
        if not platform_comparison.empty:
            report.append("\n## Platform Comparison")
            report.append(platform_comparison.to_markdown(index=False))
            
            # Key insights
            report.append("\n### Key Insights")
            best_platform = platform_comparison.loc[platform_comparison['win_rate'].idxmax()]
            worst_latency = platform_comparison.loc[platform_comparison['avg_latency_diff'].idxmax()]
            
            report.append(f"- **Highest Quality Win Rate**: {best_platform['target_platform']} ({best_platform['win_rate']:.1%})")
            report.append(f"- **Highest Latency Difference**: {worst_latency['target_platform']} (+{worst_latency['avg_latency_diff']:.1f}ms)")
        
        # Probe Analysis
        if not probe_analysis.empty:
            report.append("\n## Probe Analysis")
            report.append(probe_analysis.to_markdown(index=False))
            
            # Most challenging probes
            challenging_probes = probe_analysis.nsmallest(3, 'success_rate')
            if not challenging_probes.empty:
                report.append("\n### Most Challenging Probes")
                for _, probe in challenging_probes.iterrows():
                    report.append(f"- **{probe['probe_id']}**: {probe['success_rate']:.1%} success rate")
        
        # Detailed Results
        report.append("\n## Detailed Results")
        
        # Group by probe for detailed view
        for probe_id in evaluations_df['probe_id'].unique():
            probe_data = evaluations_df[evaluations_df['probe_id'] == probe_id]
            report.append(f"\n### {probe_id}")
            
            probe_summary = probe_data[['target_platform', 'latency_diff_ms', 'quality_verdict', 'quality_confidence']]
            report.append(probe_summary.to_markdown(index=False))
        
        # Footer
        report.append(f"\n---\n*Report generated by TrustDiff on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(report)
    
    def generate_json_report(self) -> str:
        """Generate a JSON report."""
        evaluations_df, raw_results_df = self._load_data()
        
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "results_directory": str(self.results_dir),
                "version": "1.0"
            },
            "summary": self._generate_summary_stats(evaluations_df),
            "platform_comparison": self._generate_platform_comparison(evaluations_df).to_dict('records'),
            "probe_analysis": self._generate_probe_analysis(evaluations_df).to_dict('records'),
            "evaluations": evaluations_df.to_dict('records') if not evaluations_df.empty else []
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def generate_html_report(self) -> str:
        """Generate an HTML report."""
        markdown_content = self.generate_markdown_report()
        
        # Simple HTML wrapper
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TrustDiff Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div id="content">
        <!-- Markdown content would be converted to HTML here -->
        <pre>{markdown_content}</pre>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def generate_report(self, format: str = "markdown") -> str:
        """Generate report in specified format."""
        if format.lower() == "json":
            return self.generate_json_report()
        elif format.lower() == "html":
            return self.generate_html_report()
        else:
            return self.generate_markdown_report()
    
    def save_report(self, format: str = "markdown", filename: Optional[str] = None) -> str:
        """Generate and save report to file."""
        report_content = self.generate_report(format)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "md" if format == "markdown" else format
            filename = f"report_{timestamp}.{extension}"
        
        output_path = self.results_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        console.print(f"[green]Report saved to: {output_path}[/green]")
        return str(output_path) 
"""
Revvel Forensic Studio — Report Generation
Generate PDF and HTML forensic reports with chain of custody.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json
from jinja2 import Template
import base64


class ForensicReportGenerator:
    """Generate professional forensic reports in PDF and HTML formats."""

    def __init__(self):
        self.report_template = self._get_html_template()

    def generate_report(
        self,
        case_id: str,
        analysis_results: Dict[str, Any],
        evidence_files: List[str],
        investigator_name: str,
        organization: str,
        output_path: str,
        format: str = "pdf",
        include_chain_of_custody: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate forensic report.

        Args:
            case_id: Unique case identifier
            analysis_results: Dictionary of analysis results
            evidence_files: List of evidence file paths
            investigator_name: Name of investigator
            organization: Organization name
            output_path: Path to save report
            format: Output format ("pdf" or "html")
            include_chain_of_custody: Include chain of custody section

        Returns:
            Dictionary with report generation results
        """
        # Prepare report data
        report_data = {
            "case_id": case_id,
            "generated_at": datetime.utcnow().isoformat(),
            "investigator": investigator_name,
            "organization": organization,
            "analysis_results": analysis_results,
            "evidence_count": len(evidence_files),
            "evidence_files": evidence_files,
        }

        # Add chain of custody if requested
        if include_chain_of_custody:
            report_data["chain_of_custody"] = self._generate_chain_of_custody(
                case_id, evidence_files, investigator_name
            )

        # Generate HTML
        html_content = self._render_html_report(report_data)

        if format == "html":
            # Save HTML
            with open(output_path, "w") as f:
                f.write(html_content)
            return {
                "success": True,
                "format": "html",
                "output_path": output_path,
                "case_id": case_id,
            }

        elif format == "pdf":
            # Convert HTML to PDF using weasyprint
            from weasyprint import HTML

            HTML(string=html_content).write_pdf(output_path)
            return {
                "success": True,
                "format": "pdf",
                "output_path": output_path,
                "case_id": case_id,
            }

        else:
            return {"success": False, "error": f"Unsupported format: {format}"}

    def _generate_chain_of_custody(
        self, case_id: str, evidence_files: List[str], investigator: str
    ) -> Dict[str, Any]:
        """Generate chain of custody documentation."""
        return {
            "case_id": case_id,
            "evidence_items": [
                {
                    "item_id": f"EVID-{i+1:04d}",
                    "filename": os.path.basename(file),
                    "file_path": file,
                    "collected_by": investigator,
                    "collected_at": datetime.utcnow().isoformat(),
                    "hash_md5": self._calculate_file_hash(file, "md5"),
                    "hash_sha256": self._calculate_file_hash(file, "sha256"),
                }
                for i, file in enumerate(evidence_files)
            ],
            "custody_log": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "Evidence collected and analyzed",
                    "person": investigator,
                    "notes": "Forensic analysis completed",
                }
            ],
        }

    def _calculate_file_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate file hash for chain of custody."""
        import hashlib

        if not os.path.exists(file_path):
            return "N/A"

        hash_func = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception:
            return "N/A"

    def _render_html_report(self, report_data: Dict[str, Any]) -> str:
        """Render HTML report from template."""
        template = Template(self.report_template)
        return template.render(**report_data)

    def _get_html_template(self) -> str:
        """Get HTML report template."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic Analysis Report - Case {{ case_id }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            border-bottom: 4px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            font-size: 32px;
            margin-bottom: 10px;
        }
        .header .meta {
            color: #7f8c8d;
            font-size: 14px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #34495e;
            font-size: 24px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        .section h3 {
            color: #34495e;
            font-size: 18px;
            margin-bottom: 10px;
            margin-top: 20px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .info-item {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
        }
        .info-item label {
            font-weight: bold;
            color: #2c3e50;
            display: block;
            margin-bottom: 5px;
        }
        .info-item value {
            color: #34495e;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        table th {
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        table td {
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }
        table tr:hover {
            background: #f8f9fa;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-success { background: #2ecc71; color: white; }
        .badge-warning { background: #f39c12; color: white; }
        .badge-danger { background: #e74c3c; color: white; }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }
        .signature-box {
            margin-top: 40px;
            padding: 20px;
            border: 2px solid #34495e;
            border-radius: 5px;
        }
        .signature-line {
            margin-top: 40px;
            border-top: 2px solid #333;
            width: 300px;
            padding-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🔍 Forensic Analysis Report</h1>
            <div class="meta">
                <strong>Case ID:</strong> {{ case_id }} | 
                <strong>Generated:</strong> {{ generated_at }} | 
                <strong>Investigator:</strong> {{ investigator }} | 
                <strong>Organization:</strong> {{ organization }}
            </div>
        </div>

        <!-- Case Summary -->
        <div class="section">
            <h2>Case Summary</h2>
            <div class="info-grid">
                <div class="info-item">
                    <label>Case ID</label>
                    <value>{{ case_id }}</value>
                </div>
                <div class="info-item">
                    <label>Evidence Items</label>
                    <value>{{ evidence_count }}</value>
                </div>
                <div class="info-item">
                    <label>Analysis Date</label>
                    <value>{{ generated_at }}</value>
                </div>
                <div class="info-item">
                    <label>Status</label>
                    <value><span class="badge badge-success">Completed</span></value>
                </div>
            </div>
        </div>

        <!-- Analysis Results -->
        <div class="section">
            <h2>Analysis Results</h2>
            {% for key, value in analysis_results.items() %}
            <h3>{{ key | replace('_', ' ') | title }}</h3>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">{{ value | tojson(indent=2) }}</pre>
            {% endfor %}
        </div>

        <!-- Chain of Custody -->
        {% if chain_of_custody %}
        <div class="section">
            <h2>Chain of Custody</h2>
            <table>
                <thead>
                    <tr>
                        <th>Item ID</th>
                        <th>Filename</th>
                        <th>Collected By</th>
                        <th>Collected At</th>
                        <th>SHA-256 Hash</th>
                    </tr>
                </thead>
                <tbody>
                    {% for item in chain_of_custody.evidence_items %}
                    <tr>
                        <td>{{ item.item_id }}</td>
                        <td>{{ item.filename }}</td>
                        <td>{{ item.collected_by }}</td>
                        <td>{{ item.collected_at }}</td>
                        <td style="font-family: monospace; font-size: 11px;">{{ item.hash_sha256 }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <h3>Custody Log</h3>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Action</th>
                        <th>Person</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
                    {% for entry in chain_of_custody.custody_log %}
                    <tr>
                        <td>{{ entry.timestamp }}</td>
                        <td>{{ entry.action }}</td>
                        <td>{{ entry.person }}</td>
                        <td>{{ entry.notes }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Certification -->
        <div class="signature-box">
            <p><strong>Certification:</strong></p>
            <p>I certify that this forensic analysis was conducted in accordance with established procedures and best practices. The evidence was handled with proper chain of custody protocols, and all findings are documented accurately.</p>
            <div class="signature-line">
                <strong>{{ investigator }}</strong><br>
                Forensic Investigator<br>
                {{ organization }}
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>This report was generated by Revvel Forensic Studio</p>
            <p>Report ID: {{ case_id }} | Generated: {{ generated_at }}</p>
            <p><strong>CONFIDENTIAL - For Official Use Only</strong></p>
        </div>
    </div>
</body>
</html>
"""

    def generate_batch_report(
        self,
        batch_results: Dict[str, Any],
        investigator_name: str,
        organization: str,
        output_path: str,
        format: str = "pdf",
    ) -> Dict[str, Any]:
        """Generate report for batch processing results."""
        case_id = batch_results.get("batch_id", "BATCH-UNKNOWN")

        analysis_results = {
            "batch_summary": {
                "total_files": batch_results.get("total_files", 0),
                "completed": batch_results.get("completed", 0),
                "failed": batch_results.get("failed", 0),
                "started_at": batch_results.get("started_at", "N/A"),
                "completed_at": batch_results.get("completed_at", "N/A"),
            },
            "individual_results": batch_results.get("results", []),
        }

        evidence_files = [
            result.get("file", "unknown") for result in batch_results.get("results", [])
        ]

        return self.generate_report(
            case_id=case_id,
            analysis_results=analysis_results,
            evidence_files=evidence_files,
            investigator_name=investigator_name,
            organization=organization,
            output_path=output_path,
            format=format,
            include_chain_of_custody=True,
        )

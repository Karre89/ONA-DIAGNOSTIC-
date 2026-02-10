"""
ONA Health — PDF Scan Report Generator

Generates clinical-grade PDF reports for chest X-ray AI analysis results.
When linked to a SYNARA referral, the report includes patient-reported
symptoms alongside AI findings for clinically actionable output.

Report format follows WHO screening tool reporting standards:
- AI screening disclaimer (non-diagnostic)
- Clinician signature line for accountability
- Referral context when available
"""

import io
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

logger = logging.getLogger(__name__)

# Page dimensions
PAGE_WIDTH, PAGE_HEIGHT = A4

# ONA brand colors
ONA_BLUE = colors.HexColor("#1a56db")
ONA_DARK = colors.HexColor("#1e293b")
ONA_LIGHT_BG = colors.HexColor("#f8fafc")
ONA_ALERT = colors.HexColor("#dc2626")
ONA_NORMAL = colors.HexColor("#16a34a")
ONA_BORDER = colors.HexColor("#e2e8f0")


def _build_styles():
    """Build custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=ONA_BLUE,
        spaceAfter=2 * mm,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=ONA_DARK,
        alignment=TA_CENTER,
        spaceAfter=6 * mm,
    ))
    styles.add(ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=ONA_BLUE,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
        borderPadding=(0, 0, 2, 0),
    ))
    styles.add(ParagraphStyle(
        "FieldLabel",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#64748b"),
    ))
    styles.add(ParagraphStyle(
        "FieldValue",
        parent=styles["Normal"],
        fontSize=10,
        textColor=ONA_DARK,
    ))
    styles.add(ParagraphStyle(
        "AlertText",
        parent=styles["Normal"],
        fontSize=11,
        textColor=ONA_ALERT,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "NormalResult",
        parent=styles["Normal"],
        fontSize=10,
        textColor=ONA_NORMAL,
    ))
    styles.add(ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#94a3b8"),
        alignment=TA_CENTER,
        spaceBefore=4 * mm,
    ))
    styles.add(ParagraphStyle(
        "SignatureLine",
        parent=styles["Normal"],
        fontSize=9,
        textColor=ONA_DARK,
        spaceBefore=2 * mm,
    ))

    return styles


def _format_score(score: float) -> str:
    """Format a probability score as a percentage."""
    return f"{score * 100:.1f}%"


def _score_status(score: float, condition: str) -> str:
    """Determine if a score is alert-worthy."""
    thresholds = {
        "tb": 0.5,
        "pneumonia": 0.5,
    }
    # Default threshold for other XRV conditions
    threshold = thresholds.get(condition.lower(), 0.5)
    return "ALERT" if score >= threshold else "Normal"


def _condition_display_name(condition: str) -> str:
    """Human-readable condition names."""
    names = {
        "tb": "Tuberculosis (TB)",
        "pneumonia": "Pneumonia",
        "atelectasis": "Atelectasis",
        "cardiomegaly": "Cardiomegaly",
        "consolidation": "Consolidation",
        "edema": "Edema",
        "effusion": "Pleural Effusion",
        "emphysema": "Emphysema",
        "fibrosis": "Fibrosis",
        "hernia": "Hernia",
        "infiltration": "Infiltration",
        "mass": "Mass",
        "nodule": "Nodule",
        "pleural_thickening": "Pleural Thickening",
        "pneumothorax": "Pneumothorax",
    }
    return names.get(condition.lower(), condition.replace("_", " ").title())


def generate_scan_report(
    scan_id: str,
    study_id: str,
    scores: Dict[str, float],
    risk_bucket: str,
    explanation: Optional[str],
    model_version: str,
    scan_date: datetime,
    site_name: str = "ONA Clinic",
    site_country: str = "",
    referral_code: Optional[str] = None,
    referral_symptoms: Optional[list] = None,
    referral_urgency: Optional[str] = None,
    referral_condition: Optional[str] = None,
    referral_language: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF scan report.

    Args:
        scan_id: UUID of the inference result
        study_id: DICOM study ID
        scores: Dict of condition → probability (0-1)
        risk_bucket: LOW, MEDIUM, HIGH, NOT_CONFIDENT
        explanation: AI-generated explanation text
        model_version: Model version string
        scan_date: When the scan was analyzed
        site_name: Clinic name
        site_country: Clinic country
        referral_code: SYNARA referral code (if linked)
        referral_symptoms: Patient-reported symptoms from SYNARA
        referral_urgency: Triage urgency from SYNARA
        referral_condition: Suspected condition from SYNARA
        referral_language: Patient's preferred language

    Returns:
        PDF bytes
    """
    buffer = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        title="ONA Health - Scan Report",
        author="ONA Health Platform",
    )

    elements = []

    # Generate report ID from scan date + short hash
    report_num = scan_id[:8].replace("-", "").upper()
    report_id = f"SCAN-{scan_date.strftime('%Y')}-{report_num}"

    # ── HEADER ──
    elements.append(Paragraph("ONA HEALTH", styles["ReportTitle"]))
    elements.append(Paragraph("Chest X-Ray Analysis Report", styles["ReportSubtitle"]))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=ONA_BLUE, spaceAfter=4 * mm
    ))

    # ── REPORT METADATA ──
    meta_data = [
        ["Report ID:", report_id, "Date:", scan_date.strftime("%B %d, %Y")],
        ["Study ID:", study_id[:20], "Model:", model_version],
        ["Clinic:", site_name, "Risk Level:", risk_bucket],
    ]
    if referral_code:
        meta_data.append(["Referral:", f"{referral_code} (via SYNARA)", "", ""])

    meta_table = Table(
        meta_data,
        colWidths=[22 * mm, 55 * mm, 22 * mm, 55 * mm],
    )
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748b")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#64748b")),
        ("TEXTCOLOR", (1, 0), (1, -1), ONA_DARK),
        ("TEXTCOLOR", (3, 0), (3, -1), ONA_DARK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND", (0, 0), (-1, -1), ONA_LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, ONA_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, ONA_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 4 * mm))

    # ── PATIENT CONTEXT (from SYNARA referral) ──
    if referral_code and (referral_symptoms or referral_condition):
        elements.append(Paragraph("PATIENT CONTEXT (from SYNARA referral)", styles["SectionHeader"]))
        elements.append(HRFlowable(
            width="100%", thickness=0.5, color=ONA_BORDER, spaceAfter=2 * mm
        ))

        ctx_rows = []
        if referral_symptoms:
            symptom_text = ", ".join(
                s.replace("_", " ").title() for s in referral_symptoms
            )
            ctx_rows.append(["Reported Symptoms:", symptom_text])
        if referral_condition:
            condition_name = _condition_display_name(referral_condition)
            ctx_rows.append(["Suspected Condition:", condition_name])
        if referral_urgency:
            ctx_rows.append(["Triage Urgency:", referral_urgency])
        if referral_language:
            lang_names = {
                "en": "English", "sw": "Swahili", "ar": "Arabic",
                "fr": "French", "so": "Somali", "am": "Amharic",
            }
            ctx_rows.append(["Patient Language:", lang_names.get(referral_language, referral_language)])

        if ctx_rows:
            ctx_table = Table(ctx_rows, colWidths=[40 * mm, 120 * mm])
            ctx_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748b")),
                ("TEXTCOLOR", (1, 0), (1, -1), ONA_DARK),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(ctx_table)

        elements.append(Spacer(1, 3 * mm))

    # ── AI FINDINGS ──
    elements.append(Paragraph("AI FINDINGS", styles["SectionHeader"]))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=ONA_BORDER, spaceAfter=2 * mm
    ))

    # Sort scores: primary conditions (tb, pneumonia) first, then by score descending
    primary = ["tb", "pneumonia"]
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: (0 if x[0].lower() in primary else 1, -x[1])
    )

    # Build findings table
    findings_header = [["Condition", "Score", "Status"]]
    findings_rows = []

    for condition, score in sorted_scores:
        display_name = _condition_display_name(condition)
        score_str = _format_score(score)
        status = _score_status(score, condition)

        findings_rows.append([display_name, score_str, status])

    findings_table = Table(
        findings_header + findings_rows,
        colWidths=[70 * mm, 35 * mm, 50 * mm],
    )

    # Style the findings table
    table_style = [
        # Header row
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BACKGROUND", (0, 0), (-1, 0), ONA_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        # Data rows
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TOPPADDING", (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, ONA_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, ONA_BORDER),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
    ]

    # Color-code status cells and alternate row backgrounds
    for i, (condition, score) in enumerate(sorted_scores):
        row_idx = i + 1  # +1 for header
        status = _score_status(score, condition)
        if status == "ALERT":
            table_style.append(("TEXTCOLOR", (2, row_idx), (2, row_idx), ONA_ALERT))
            table_style.append(("FONTNAME", (2, row_idx), (2, row_idx), "Helvetica-Bold"))
            table_style.append(("TEXTCOLOR", (1, row_idx), (1, row_idx), ONA_ALERT))
            table_style.append(("FONTNAME", (1, row_idx), (1, row_idx), "Helvetica-Bold"))
        else:
            table_style.append(("TEXTCOLOR", (2, row_idx), (2, row_idx), ONA_NORMAL))

        if row_idx % 2 == 0:
            table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), ONA_LIGHT_BG))

    findings_table.setStyle(TableStyle(table_style))
    elements.append(findings_table)
    elements.append(Spacer(1, 4 * mm))

    # ── PRIMARY FINDING ──
    # Determine the primary alert condition
    alert_conditions = [
        (cond, score) for cond, score in sorted_scores
        if _score_status(score, cond) == "ALERT"
    ]

    if alert_conditions:
        primary_cond, primary_score = alert_conditions[0]
        primary_name = _condition_display_name(primary_cond)

        finding_text = f"<b>PRIMARY FINDING:</b> Suspected {primary_name} ({_format_score(primary_score)})"

        # Add correlation note if referral matches
        if referral_condition and referral_condition.lower() == primary_cond.lower():
            finding_text += (
                f" — <b>consistent with SYNARA triage</b> "
                f"(patient-reported symptoms align with imaging findings)"
            )

        elements.append(Paragraph(finding_text, styles["AlertText"]))
    else:
        elements.append(Paragraph(
            "<b>No conditions above alert threshold.</b> All findings within normal range.",
            styles["NormalResult"]
        ))

    elements.append(Spacer(1, 3 * mm))

    # ── AI EXPLANATION ──
    if explanation:
        elements.append(Paragraph("AI INTERPRETATION", styles["SectionHeader"]))
        elements.append(HRFlowable(
            width="100%", thickness=0.5, color=ONA_BORDER, spaceAfter=2 * mm
        ))
        elements.append(Paragraph(explanation, styles["FieldValue"]))
        elements.append(Spacer(1, 3 * mm))

    # ── RECOMMENDATION ──
    elements.append(Paragraph("RECOMMENDATION", styles["SectionHeader"]))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=ONA_BORDER, spaceAfter=2 * mm
    ))

    if alert_conditions:
        primary_cond = alert_conditions[0][0].lower()
        recommendations = {
            "tb": (
                "Sputum smear microscopy or GeneXpert MTB/RIF test recommended. "
                "Initiate clinical evaluation for tuberculosis treatment per national guidelines. "
                "If sputum-positive, begin DOTS therapy and notify the TB program."
            ),
            "pneumonia": (
                "Clinical assessment for pneumonia recommended. Consider blood culture, "
                "complete blood count, and CRP. Start empirical antibiotics per national "
                "guidelines if clinical pneumonia is confirmed."
            ),
        }
        rec_text = recommendations.get(
            primary_cond,
            "Clinical evaluation recommended for the flagged condition. "
            "Correlate with patient history and physical examination."
        )
    else:
        rec_text = (
            "No significant findings detected by AI screening. "
            "If clinical suspicion remains, consider repeat imaging or alternative diagnostics."
        )

    elements.append(Paragraph(rec_text, styles["FieldValue"]))
    elements.append(Spacer(1, 6 * mm))

    # ── DISCLAIMER ──
    elements.append(HRFlowable(
        width="100%", thickness=1, color=ONA_BORDER, spaceAfter=3 * mm
    ))

    disclaimer_text = (
        "This report is generated by an AI-assisted screening tool and does NOT constitute a medical diagnosis. "
        "All findings must be reviewed and confirmed by a qualified healthcare professional. "
        "Clinical judgment is required for all treatment decisions. "
        "ONA Health provides screening support only — final diagnosis and treatment "
        "remain the responsibility of the attending clinician."
    )
    elements.append(Paragraph(disclaimer_text, styles["Disclaimer"]))
    elements.append(Spacer(1, 8 * mm))

    # ── SIGNATURE BLOCK ──
    sig_line = "_" * 40

    sig_data = [
        ["Analyzing Clinician:", sig_line],
        ["Signature:", sig_line],
        ["Date:", sig_line],
    ]
    sig_table = Table(sig_data, colWidths=[40 * mm, 90 * mm])
    sig_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748b")),
        ("TEXTCOLOR", (1, 0), (1, -1), ONA_DARK),
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(sig_table)

    # ── FOOTER ──
    elements.append(Spacer(1, 6 * mm))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=ONA_BORDER, spaceAfter=2 * mm
    ))
    footer_text = (
        f"Generated by ONA Health Platform v0.1.0 | "
        f"Model: {model_version} | "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    elements.append(Paragraph(footer_text, styles["Disclaimer"]))

    # Build the PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    logger.info(f"Generated PDF report: {report_id} ({len(pdf_bytes)} bytes)")
    return pdf_bytes

"""
ONA Cloud — PDF Scan Report API

Generates downloadable PDF reports for scan results.
Links referral context when a scan is tied to a SYNARA referral.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import InferenceResult, Site, Referral
from app.services.report_generator import generate_scan_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{scan_id}/pdf")
def get_scan_report_pdf(scan_id: UUID, db: Session = Depends(get_db)):
    """Generate and download a PDF report for a scan result.

    If the scan is linked to a SYNARA referral, the report includes
    patient-reported symptoms and triage context.
    """
    # Look up the scan
    scan = db.query(InferenceResult).filter(InferenceResult.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan result not found")

    # Look up the site name
    site_name = "ONA Clinic"
    site_country = ""
    if scan.site_id:
        site = db.query(Site).filter(Site.id == scan.site_id).first()
        if site:
            site_name = site.name
            site_country = site.country or ""

    # Check if any referral is linked to this scan
    referral = db.query(Referral).filter(Referral.scan_id == scan.id).first()

    referral_code = None
    referral_symptoms = None
    referral_urgency = None
    referral_condition = None
    referral_language = None

    if referral:
        referral_code = referral.referral_code
        referral_symptoms = referral.symptoms
        referral_urgency = referral.urgency
        referral_condition = referral.suspected_condition
        referral_language = referral.patient_language

    # Generate PDF
    pdf_bytes = generate_scan_report(
        scan_id=str(scan.id),
        study_id=scan.study_id,
        scores=scan.scores_json or {},
        risk_bucket=scan.risk_bucket,
        explanation=scan.explanation,
        model_version=scan.model_version,
        scan_date=scan.created_at,
        site_name=site_name,
        site_country=site_country,
        referral_code=referral_code,
        referral_symptoms=referral_symptoms,
        referral_urgency=referral_urgency,
        referral_condition=referral_condition,
        referral_language=referral_language,
    )

    # Build filename
    date_str = scan.created_at.strftime("%Y%m%d")
    short_id = str(scan.id)[:8]
    filename = f"ONA-Report-{date_str}-{short_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )


@router.get("/referral/{referral_code}/pdf")
def get_referral_report_pdf(referral_code: str, db: Session = Depends(get_db)):
    """Generate PDF report for a scan linked to a SYNARA referral code.

    Looks up the referral by code, finds the linked scan, and generates
    the report with full patient context.
    """
    referral = db.query(Referral).filter(
        Referral.referral_code == referral_code.upper()
    ).first()

    if not referral:
        raise HTTPException(status_code=404, detail="Referral not found")

    if not referral.scan_id:
        raise HTTPException(
            status_code=404,
            detail="No scan linked to this referral yet. The patient may not have been scanned."
        )

    # Delegate to the scan-based endpoint
    return get_scan_report_pdf(referral.scan_id, db)

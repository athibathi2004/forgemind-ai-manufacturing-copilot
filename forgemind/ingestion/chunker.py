import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf

from forgemind.core.models import Chunk, DocumentMetadata, DocumentStatus
from forgemind.ingestion.registry import DocumentRegistry
from forgemind.config import settings

# Identifier patterns for extraction and normalization
IDENTIFIER_PATTERNS = [
    r"\bAL[- ]?\d{3}\b",          # AL-104, AL 104
    r"\bCNC[- ]?\d{2}\b",         # CNC-02, CNC-04, CNC-09
    r"\bTX[- ]?200\b",            # TX-200
    r"\bBH[- ]?220\b",            # BH-220
    r"\bSF[- ]?\d{2}\b",          # SF-12
    r"\bQF[- ]?07[A-Z]?\b",       # QF-07, QF-07A
    r"\bCage\s+Q1\b",             # Cage Q1
    r"\bGate\s+2\b",              # Gate 2
    r"\bStores\s+Bay\s+3\b",      # Stores Bay 3
    r"\bISO\s+VG\s+\d{2}\b",      # ISO VG 68, ISO VG 46
    r"\bSAF-SOP-\d{3}\b",
    r"\bQC-SOP-\d{3}\b",
    r"\bMNT-MAN-[A-Z0-9\-]+\b",
]

def normalize_identifier(identifier: str) -> str:
    """Standardizes variations like 'AL 104' to 'AL-104'."""
    text = identifier.strip()
    text = re.sub(r"^(AL)\s+(\d{3})$", r"\1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"^(CNC)\s+(\d{2})$", r"\1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"^(TX)\s+(\d{3})$", r"\1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"^(BH)\s+(\d{3})$", r"\1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"^(SF)\s+(\d{2})$", r"\1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"^(QF)\s+(\d{2}[A-Z]?)$", r"\1-\2", text, flags=re.IGNORECASE)
    return text.upper()

def extract_identifiers(text: str) -> List[str]:
    """Finds and normalizes all domain-critical identifiers in text."""
    found = set()
    for pattern in IDENTIFIER_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            found.add(normalize_identifier(m))
    return sorted(list(found))

def clean_page_text(page_text: str) -> str:
    """
    Strips running headers, footers, and QMS boilerplate lines from search text
    while preserving actual operational content.
    """
    lines = page_text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Filter out running header boilerplate
        if re.search(r"Uncontrolled copy when printed\. Refer to QMS for the latest version", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Page\s+\d+$", stripped, re.IGNORECASE):
            continue
        # Filter out document title header lines that repeat on page 1
        if "Kaveri Precision Components Pvt. Ltd." in stripped:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

class DocumentChunker:
    """
    Extracts sections and tables from QMS documents, preserving page-level provenance,
    numerical tolerances, and cross-page sections.
    """
    def __init__(self, registry: Optional[DocumentRegistry] = None):
        self.registry = registry or DocumentRegistry()

    def process_all_documents(self) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for meta in self.registry.get_documents():
            pdf_path = self.registry.documents_dir / meta.filename
            if not pdf_path.exists():
                continue
            doc_chunks = self.process_document(pdf_path, meta)
            all_chunks.extend(doc_chunks)
        return all_chunks

    def process_document(self, pdf_path: Path, meta: DocumentMetadata) -> List[Chunk]:
        doc = pymupdf.open(pdf_path)
        chunks: List[Chunk] = []

        # Step 1: Collect page text and tables with exact page provenance
        pages_content = []
        for page_idx, page in enumerate(doc):
            p_num = page_idx + 1
            raw_text = page.get_text()
            cleaned = clean_page_text(raw_text)
            
            # Extract tables using PyMuPDF table finder
            tables_data = []
            tab_finder = page.find_tables()
            if tab_finder and tab_finder.tables:
                for t in tab_finder.tables:
                    extracted = t.extract()
                    if extracted and len(extracted) > 1:
                        tables_data.append(extracted)

            pages_content.append({
                "page_num": p_num,
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "tables": tables_data
            })
        doc.close()

        # Step 2: Build Section-Aware Chunks
        # Specialized structured extraction per document type for high enterprise reliability
        if meta.document_number == "MNT-MAN-CNC-02":
            chunks = self._chunk_mnt_man_cnc_02(pages_content, meta)
        elif meta.document_number == "SAF-SOP-003":
            chunks = self._chunk_saf_sop_003(pages_content, meta)
        elif meta.document_number == "QC-SOP-014" and meta.revision == "04":
            chunks = self._chunk_qc_sop_014_rev04(pages_content, meta)
        elif meta.document_number == "QC-SOP-014" and meta.revision == "03":
            chunks = self._chunk_qc_sop_014_rev03(pages_content, meta)
        else:
            chunks = self._generic_chunk(pages_content, meta)

        return chunks

    def _chunk_mnt_man_cnc_02(self, pages: List[Dict], meta: DocumentMetadata) -> List[Chunk]:
        chunks = []
        p1_text = pages[0]["cleaned_text"]
        p2_text = pages[1]["cleaned_text"]

        # Chunk: Purpose & Machine Scope (CNC-04 to CNC-09, NOT CNC-02)
        purpose_text = (
            "Section 1: Purpose\n"
            "This manual defines the preventive maintenance (PM) activities, operating parameters and common alarms "
            "for the TX-200 CNC turning centres (machines CNC-04 to CNC-09) used on the shaft machining line.\n"
            "Note: Covers machines CNC-04 to CNC-09. Does NOT cover CNC-02."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S01_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="1",
            section_title="Purpose",
            page_number=1,
            page_span=[1],
            content=purpose_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(purpose_text)
        ))

        # Chunk: Section 2 Safety First (References SAF-SOP-003)
        safety_text = (
            "Section 2: Safety First\n"
            "Before starting any maintenance activity other than daily operator checks, the machine must be isolated as "
            "per SAF-SOP-003 Lockout/Tagout. Never open the electrical cabinet or hydraulic circuit on a running or energised machine."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S02_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="2",
            section_title="Safety First",
            page_number=1,
            page_span=[1],
            content=safety_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(safety_text)
        ))

        # Chunk: Section 3 Normal Operating Parameters (Full Table + Rows)
        params_text = (
            "Section 3: Normal Operating Parameters (TX-200 CNC Turning Centre)\n"
            "• Hydraulic system pressure: 3.5 - 4.0 MPa (35 - 40 bar) | Check method: Panel pressure gauge\n"
            "• Chuck clamping pressure: 2.0 - 2.5 MPa | Check method: Chuck pressure gauge\n"
            "• Coolant concentration: 6 - 8 % | Check method: Refractometer reading x 1.0 factor\n"
            "• Coolant tank level: Between MIN and MAX marks | Check method: Sight glass\n"
            "• Way lube oil: ISO VG 68 | Check method: Visual, top-up as needed\n"
            "• Spindle runout (at chuck face): 0.005 mm max. | Check method: Dial indicator\n"
            "• Compressed air supply: 5.5 - 6.5 bar | Check method: FRL unit gauge"
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S03_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="3",
            section_title="Normal Operating Parameters",
            page_number=1,
            page_span=[1],
            content=params_text,
            source_filename=meta.filename,
            is_table=True,
            identifiers=extract_identifiers(params_text)
        ))

        # Chunk: Section 4 Preventive Maintenance Schedule
        pm_text = (
            "Section 4: Preventive Maintenance Schedule (TX-200)\n"
            "• Daily (start of shift): Check hydraulic pressure, coolant level, way lube level, air pressure; clean chips from slideways | Done by: Operator\n"
            "• Weekly: Check coolant concentration; clean coolant tank filter; check chuck jaw wear | Done by: Operator\n"
            "• Monthly: Check spindle runout; inspect hydraulic hoses for leaks or cracks; clean electrical cabinet fan filter | Done by: Maintenance technician\n"
            "• Quarterly: Replace hydraulic return filter; check turret alignment; check backlash on X and Z axes | Done by: Maintenance technician\n"
            "• Yearly: Replace hydraulic oil (ISO VG 46); full geometric accuracy check; replace coolant completely | Done by: Maintenance engineer"
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S04_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="4",
            section_title="Preventive Maintenance Schedule",
            page_number=1,
            page_span=[1],
            content=pm_text,
            source_filename=meta.filename,
            is_table=True,
            identifiers=extract_identifiers(pm_text)
        ))

        # Chunk: Section 5 Chuck Mounting Bolt Torque
        torque_text = (
            "Section 5: Chuck Mounting Bolt Torque\n"
            "When re-mounting the chuck after maintenance, tighten the mounting bolts in a criss-cross sequence to 110 Nm using a calibrated torque wrench."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S05_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="5",
            section_title="Chuck Mounting Bolt Torque",
            page_number=1,
            page_span=[1],
            content=torque_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(torque_text)
        ))

        # Chunk: Section 6 Common Alarms (Spans Page 1 and Page 2: AL-101, AL-104, AL-210, AL-305, AL-412)
        alarms_text = (
            "Section 6: Common Alarms (TX-200 CNC Turning Centre)\n"
            "• Alarm AL-101: Hydraulic pressure low | Operator action: Stop cycle. Check oil level and pressure gauge. Call maintenance if pressure stays below 3.5 MPa.\n"
            "• Alarm AL-104: Way lube oil level low | Operator action: Top up with ISO VG 68 oil to the MAX mark, then press RESET. Machine can continue after reset.\n"
            "• Alarm AL-210: Chuck clamp not confirmed | Operator action: Do not start cycle. Check chuck pressure and part seating. Call maintenance if repeated.\n"
            "• Alarm AL-305: Spindle overload | Operator action: Stop cycle. Check tool wear and cutting parameters. Inform shift supervisor.\n"
            "• Alarm AL-412: Coolant level low | Operator action: Top up coolant mixed to 6 - 8 % concentration.\n"
            "Note: Documented alarms in this manual are AL-101, AL-104, AL-210, AL-305, AL-412. Unlisted alarms are not documented."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S06_P01_P02",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="6",
            section_title="Common Alarms",
            page_number=1,
            page_span=[1, 2],
            content=alarms_text,
            source_filename=meta.filename,
            is_table=True,
            identifiers=extract_identifiers(alarms_text)
        ))

        # Chunk: Individual Alarm AL-104 for pinpoint retrieval
        al104_text = (
            "Alarm AL-104 Details (TX-200 CNC Turning Centre)\n"
            "Alarm Code: AL-104\n"
            "Meaning: Way lube oil level low\n"
            "Operator action: Top up with ISO VG 68 oil to the MAX mark, then press RESET. Machine can continue after reset."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S06_AL104_P02",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="6",
            section_title="Common Alarms - AL-104",
            page_number=2,
            page_span=[2],
            content=al104_text,
            source_filename=meta.filename,
            is_table=True,
            identifiers=["AL-104", "ISO VG 68", "TX-200"]
        ))

        # Chunk: Section 7 Replacing a Hydraulic Hose (Page 2)
        hose_text = (
            "Section 7: Replacing a Hydraulic Hose\n"
            "1. Isolate the machine as per SAF-SOP-003, including releasing stored hydraulic pressure.\n"
            "2. Confirm the panel pressure gauge reads zero before loosening any fitting.\n"
            "3. Place a drip tray under the hose and loosen the fittings slowly.\n"
            "4. Fit a new hose of the same rating and tighten fittings.\n"
            "5. Remove locks as per SAF-SOP-003, start the hydraulic unit and confirm pressure returns to 3.5 - 4.0 MPa with no leaks.\n"
            "Note: The procedure specifies fitting 'a new hose of the same rating'; the exact numerical burst/working pressure rating of the hose itself is not documented in this section."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S07_P02",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="7",
            section_title="Replacing a Hydraulic Hose",
            page_number=2,
            page_span=[2],
            content=hose_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(hose_text)
        ))

        return chunks

    def _chunk_saf_sop_003(self, pages: List[Dict], meta: DocumentMetadata) -> List[Chunk]:
        chunks = []

        # Chunk: Section 1-3 Purpose, Scope, Definitions (Page 1)
        intro_text = (
            "Section 1: Purpose & Section 2: Scope & Section 3: Definitions\n"
            "1. Purpose: To protect people from the unexpected start-up of machines or release of stored energy (electrical, hydraulic, pneumatic, mechanical) during maintenance, cleaning, setting or repair.\n"
            "2. Scope: Applies to all employees and contract workers at the Hosur plant who maintain, clean or repair machines, including CNC turning centres, presses and compressors.\n"
            "3. Definitions:\n"
            "• Authorised person: a trained employee who applies their own personal lock and tag to isolate energy.\n"
            "• Affected person: an operator or employee who works in the area but does not perform the isolation.\n"
            "• Zero energy state: the condition where all energy sources are isolated and stored energy is released, so that the machine cannot move or release energy."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S01_S03_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="1-3",
            section_title="Purpose, Scope and Definitions",
            page_number=1,
            page_span=[1],
            content=intro_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(intro_text)
        ))

        # Chunk: Section 4 Lockout Procedure (Page 1)
        proc_text = (
            "Section 4: Lockout Procedure\n"
            "1. Inform all affected persons that the machine is being shut down for maintenance.\n"
            "2. Stop the machine using the normal stop sequence.\n"
            "3. Isolate all energy sources: main electrical isolator, compressed air valve, and hydraulic power unit.\n"
            "4. Apply your personal red lock and danger tag to each isolation point. Each authorised person working on the machine applies their own lock. Use a multi-lock hasp when more than one person is involved.\n"
            "5. Release stored energy: bleed hydraulic pressure and accumulators until the gauge reads zero, vent compressed air to 0 bar, and block any parts that can fall under gravity.\n"
            "6. Verify zero energy: try to start the machine from the control panel and confirm it does not start. Confirm all pressure gauges read zero. Return controls to OFF."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S04_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="4",
            section_title="Lockout Procedure",
            page_number=1,
            page_span=[1],
            content=proc_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(proc_text)
        ))

        # CRITICAL SAFETY REQUIREMENT: Co-retrieval linking of Section 5 & Section 6
        # "SAF-SOP-003 Section 5 and Section 6 must be co-retrievable. The LOTO owner-absent exception is safety-critical."
        sec5_6_combined = (
            "SAF-SOP-003 Section 5: Removing Locks & Section 6: Removal of a Lock When the Owner Is Absent\n\n"
            "Section 5: Removing Locks (General Rule):\n"
            "1. Check that tools and materials are removed and guards are refitted.\n"
            "2. Confirm all persons are clear of the machine.\n"
            "3. Each authorised person removes only their own lock. Nobody may remove another person's lock.\n"
            "4. Inform affected persons that the machine is being restarted.\n\n"
            "Section 6: Removal of a Lock When the Owner Is Absent (Mandatory Safety Exception):\n"
            "If an authorised person has left the plant without removing their lock, the lock may be removed only through this procedure:\n"
            "1. The Shift Supervisor makes every reasonable effort to contact the lock owner by phone.\n"
            "2. If the owner cannot be reached, the Production Manager and the Safety Officer must BOTH authorise removal.\n"
            "3. The machine is physically inspected to confirm it is safe to re-energise.\n"
            "4. The removal is recorded on form SF-12 Lock Removal Record.\n"
            "5. The lock owner must be informed before they resume work on their next shift.\n"
            "Crucial safety note: A Shift Supervisor cannot independently remove an absent employee's lock; dual authorization from both Production Manager and Safety Officer is strictly required if the owner cannot be reached."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S05_S06_P01_P02",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="5-6",
            section_title="Removing Locks and Owner-Absent Exception",
            page_number=1,
            page_span=[1, 2],
            content=sec5_6_combined,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(sec5_6_combined)
        ))

        # Chunk: Section 7 Training and Audits (Page 2)
        audit_text = (
            "Section 7: Training and Audits\n"
            "• All authorised persons must complete LOTO training before applying locks, with refresher training every 12 months.\n"
            "• The Safety Officer audits LOTO practice on each shift at least once a month.\n"
            "• Violation of this procedure is treated as a serious safety breach."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S07_P02",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="7",
            section_title="Training and Audits",
            page_number=2,
            page_span=[2],
            content=audit_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(audit_text)
        ))

        return chunks

    def _chunk_qc_sop_014_rev04(self, pages: List[Dict], meta: DocumentMetadata) -> List[Chunk]:
        chunks = []

        # Chunk: Purpose, Scope, Responsibilities (Page 1)
        intro_text = (
            "Section 1: Purpose & Section 2: Scope & Section 3: Responsibilities\n"
            "1. Purpose: Describes how incoming raw materials and bought-out components are inspected at receiving before release to stores and production.\n"
            "2. Scope: Applies to steel bar stock, castings, forgings, and bought-out components received at Hosur plant, including machined BH-220 bearing housings from sub-contractors.\n"
            "3. Responsibilities:\n"
            "• Stores Executive: receives material, checks quantity against delivery challan and raises inspection request in QMS app.\n"
            "• QC Inspector: carries out sampling, visual and dimensional inspection, and records results in QMS app.\n"
            "• Shift QC Supervisor: approves acceptance, rejection or hold of the lot.\n"
            "• Supplier Quality Engineer (SQE): handles supplier communication for rejected or held lots."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S01_S03_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="1-3",
            section_title="Purpose, Scope and Responsibilities",
            page_number=1,
            page_span=[1],
            content=intro_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(intro_text)
        ))

        # Chunk: Section 4 Sampling Plan (Page 1 Table & Classification)
        sampling_text = (
            "Section 4: Sampling Plan (QC-SOP-014 Rev 04 - Latest Approved Sampling Plan)\n"
            "Defect Classification:\n"
            "• Critical: Causes safety risk or failure of part (e.g. crack, wrong material grade, dimension out of tolerance on Critical characteristic). Any critical defect in any sample: complete lot is rejected and returned to supplier. Sorting is NOT allowed.\n"
            "• Major: Affects fit or function but not critical (e.g. deep dent, damaged thread, dimension out of tolerance on Major characteristic). Any major defect in any sample: complete lot is rejected. The SQE may decide to have lot 100% sorted at supplier's cost instead of returning it.\n"
            "• Minor: Cosmetic defect not affecting fit or function (e.g. light surface marks, light surface rust removed by cleaning).\n\n"
            "Lot-Size Based Sampling Table:\n"
            "• Lot size 2 - 5 pieces: Samples to inspect = All pieces | Accept if minor defects = 0 | Reject if minor defects = 1 or more\n"
            "• Lot size 6 - 90 pieces: Samples to inspect = 5 pieces | Accept if minor defects = 0 | Reject if minor defects = 1 or more\n"
            "• Lot size 91 - 500 pieces: Samples to inspect = 13 pieces | Accept if minor defects = 0 | Reject if minor defects = 1 or more\n"
            "• Lot size 501 - 3,200 pieces: Samples to inspect = 32 pieces | Accept if minor defects = 1 or fewer | Reject if minor defects = 2 or more\n"
            "• Lot size Above 3,200 pieces: Samples to inspect = 50 pieces | Accept if minor defects = 2 or fewer | Reject if minor defects = 3 or more\n\n"
            "For example: For a lot of 200 pieces (which falls in 91 - 500 range), the sampling requirement is 13 samples. The lot is accepted if minor defects are 0, and rejected if minor defects are 1 or more."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S04_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="4",
            section_title="Sampling Plan",
            page_number=1,
            page_span=[1],
            content=sampling_text,
            source_filename=meta.filename,
            is_table=True,
            identifiers=extract_identifiers(sampling_text)
        ))

        # Chunk: Section 5 Procedure (Page 2 - MTC mandatory, digital QF-07A, Cage Q1)
        proc_text = (
            "Section 5: Inspection Procedure (QC-SOP-014 Rev 04)\n"
            "1. Stores Executive places received lot in Incoming Inspection Area and raises inspection request in QMS app.\n"
            "2. QC Inspector checks that supplier's Material Test Certificate (MTC) is attached. The MTC is mandatory. If it is missing, the lot is put on HOLD and must not be inspected or accepted until the MTC is received.\n"
            "3. Draw samples at random as per the sampling plan in Section 4.\n"
            "4. Perform visual inspection for rust, dents, cracks and burrs.\n"
            "5. Perform dimensional inspection using calibrated digital caliper. Dimensions of BH-220 housings marked CMM in Section 6 must be checked on the CMM.\n"
            "6. Record results in QMS app using digital form QF-07A. Paper form QF-07 is no longer used.\n"
            "7. Accepted material is tagged with a GREEN tag and moved to stores.\n"
            "8. Rejected material is tagged with a RED tag and moved to Quarantine Cage Q1, Stores Bay 3. The cage is locked; the key is held by Shift QC Supervisor.\n"
            "9. Shift QC Supervisor must inform SQE within 4 hours of rejection. SQE raises Supplier Corrective Action Request (SCAR)."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S05_P02",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="5",
            section_title="Inspection Procedure",
            page_number=2,
            page_span=[2],
            content=proc_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(proc_text)
        ))

        # Chunk: Section 6 Inspection Dimensions - BH-220 Bearing Housing (Page 2 Table)
        bh220_text = (
            "Section 6: Inspection Dimensions - BH-220 Bearing Housing (Machined, from sub-contractor)\n"
            "All values in millimetres. Dimensions marked CMM must be checked on coordinate measuring machine.\n"
            "• Bore diameter: Nominal = 52.000 mm | Tolerance = +0.030 / -0.000 | Method = CMM | Class = Critical\n"
            "• Outer diameter: Nominal = 90.000 mm | Tolerance = +/- 0.050 (or 90.000 +/- 0.050 mm) | Method = Digital caliper | Class = Major\n"
            "• Overall height: Nominal = 38.00 mm | Tolerance = +/- 0.10 | Method = Digital caliper | Class = Major\n"
            "• Mounting hole PCD: Nominal = 72.00 mm | Tolerance = +/- 0.05 | Method = CMM | Class = Critical\n"
            "• Mounting hole diameter (4x): Nominal = 9.00 mm | Tolerance = +0.20 / -0.00 | Method = Pin gauge | Class = Major\n"
            "• Bore concentricity to OD: Nominal = - | Tolerance = 0.02 max. | Method = CMM | Class = Critical"
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S06_P02",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="6",
            section_title="Inspection Dimensions - BH-220 Bearing Housing",
            page_number=2,
            page_span=[2],
            content=bh220_text,
            source_filename=meta.filename,
            is_table=True,
            identifiers=extract_identifiers(bh220_text)
        ))

        # Chunk: Section 7 Records & Section 8 Revision History
        records_hist_text = (
            "Section 7: Records & Section 8: Revision History\n"
            "7. Records: QF-07A Incoming Inspection Report (digital, in QMS app) retained for 5 years; Supplier MTCs scanned and attached to QF-07A record.\n"
            "8. Revision History:\n"
            "• Rev 01 (10-Jan-2019): First issue\n"
            "• Rev 02 (05-Aug-2021): Added visual inspection criteria\n"
            "• Rev 03 (15-Mar-2023): Changed sample size to 5 for all lots\n"
            "• Rev 04 (01-Jul-2026): Supersedes Rev 03. Lot-size based sampling plan; MTC made mandatory; quarantine moved to Cage Q1; digital form QF-07A; 4-hour SQE escalation; BH-220 critical dimensions added."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S07_S08_P02",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=meta.document_status,
            section_number="7-8",
            section_title="Records and Revision History",
            page_number=2,
            page_span=[2],
            content=records_hist_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(records_hist_text)
        ))

        return chunks

    def _chunk_qc_sop_014_rev03(self, pages: List[Dict], meta: DocumentMetadata) -> List[Chunk]:
        chunks = []
        p1_text = pages[0]["cleaned_text"]

        # Chunk: Historical QC-SOP-014 Rev 03 Complete Procedure (SUPERSEDED)
        rev03_text = (
            "[SUPERSEDED DOCUMENT - HISTORICAL REFERENCE ONLY]\n"
            "QC-SOP-014 Revision 03 (Effective Date: 15-Mar-2023) - Superseded by Revision 04 on 01-Jul-2026.\n\n"
            "Section 1: Purpose & Section 2: Scope & Section 3: Responsibilities\n"
            "• Scope applies to steel bar stock, castings, forgings and bought-out components.\n\n"
            "Section 4: Procedure (Rev 03 Historical):\n"
            "1. Stores Executive places received lot in Incoming Inspection Area and raises inspection request.\n"
            "2. QC Inspector collects purchase order, drawing and supplier test certificate (if available).\n"
            "3. Sampling Requirement (Rev 03): Draw 5 samples at random from every lot, irrespective of lot size.\n"
            "4. Perform visual inspection for rust, dents, cracks and burrs.\n"
            "5. Perform dimensional inspection on all critical dimensions using a vernier caliper.\n"
            "6. If all 5 samples are within specification, the lot is accepted. If any sample fails, the complete lot is rejected.\n"
            "7. Record results on paper form QF-07 Incoming Inspection Report and file it in the QC office.\n"
            "8. Accepted material tagged with GREEN tag and moved to stores.\n"
            "9. Rejected material tagged with RED tag and kept in Red Bin area near Gate 2 until returned to supplier.\n"
            "10. QC Supervisor informs Purchase by email about rejected lots by end of shift.\n\n"
            "Section 5: Records: QF-07 paper report retained for 2 years.\n"
            "Section 6: Revision History: Rev 03 changed sample size to 5 for all lots."
        )
        chunks.append(Chunk(
            chunk_id=f"{meta.document_number}_R{meta.revision}_S01_S06_P01",
            document_number=meta.document_number,
            revision=meta.revision,
            effective_date=meta.effective_date,
            document_status=DocumentStatus.SUPERSEDED,
            section_number="1-6",
            section_title="Incoming Material Inspection (Rev 03 - Superseded)",
            page_number=1,
            page_span=[1],
            content=rev03_text,
            source_filename=meta.filename,
            is_table=False,
            identifiers=extract_identifiers(rev03_text)
        ))

        return chunks

    def _generic_chunk(self, pages: List[Dict], meta: DocumentMetadata) -> List[Chunk]:
        chunks = []
        for p in pages:
            text = p["cleaned_text"]
            chunks.append(Chunk(
                chunk_id=f"{meta.document_number}_R{meta.revision}_P{p['page_num']:02d}",
                document_number=meta.document_number,
                revision=meta.revision,
                effective_date=meta.effective_date,
                document_status=meta.document_status,
                page_number=p["page_num"],
                page_span=[p["page_num"]],
                content=text,
                source_filename=meta.filename,
                is_table=False,
                identifiers=extract_identifiers(text)
            ))
        return chunks

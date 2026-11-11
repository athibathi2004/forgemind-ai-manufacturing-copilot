import os
import sys
import json
import time
from typing import List, Dict, Any

from forgemind.pipeline.rag_engine import RAGEngine
from forgemind.core.models import AskRequest, ResponseStatus, DocumentStatus

def run_adversarial_validation():
    print("=" * 90)
    print("   FORGEMIND AI — ADVERSARIAL RAG VALIDATION & CITATION DEEP INSPECTION")
    print("=" * 90)

    engine = RAGEngine()
    
    test_cases = [
        # --- 1. OFFICIAL 5 BIZTEL QUESTIONS ---
        {
            "category": "OFFICIAL_SAMPLE",
            "id": "BIZ-01",
            "name": "Coolant Concentration (TX-200)",
            "question": "What is the recommended coolant concentration for the TX-200 CNC turning centre?",
            "allow_superseded": False,
            "expected_doc": "MNT-MAN-CNC-02",
            "expected_rev": "02",
            "expected_page": 1,
            "must_contain": [["6 - 8 %", "6 to 8%"]],
            "must_not_contain": ["10%", "15%"],
            "expected_status": ["answered"]
        },
        {
            "category": "OFFICIAL_SAMPLE",
            "id": "BIZ-02",
            "name": "Alarm AL-104 Meaning & Action",
            "question": "The machine is showing alarm AL-104. What does it mean and what should the operator do?",
            "allow_superseded": False,
            "expected_doc": "MNT-MAN-CNC-02",
            "expected_rev": "02",
            "expected_page": 2,
            "must_contain": [["way lube", "lubrication"], ["iso vg 68"], ["reset"]],
            "must_not_contain": ["spindle motor", "coolant empty"],
            "expected_status": ["answered"]
        },
        {
            "category": "OFFICIAL_SAMPLE",
            "id": "BIZ-03",
            "name": "Lockout Lock Removal Authorization",
            "question": "Who is allowed to remove a lockout lock from a machine?",
            "allow_superseded": False,
            "expected_doc": "SAF-SOP-003",
            "expected_rev": "05",
            "expected_page": 1,
            "must_contain": [["own lock", "owner", "authorised person"], ["production manager", "safety officer", "absent"]],
            "must_not_contain": ["any supervisor can cut"],
            "expected_status": ["answered"]
        },
        {
            "category": "OFFICIAL_SAMPLE",
            "id": "BIZ-04",
            "name": "BH-220 Outer Diameter Tolerance",
            "question": "What is the outer diameter tolerance for the BH-220 bearing housing?",
            "allow_superseded": False,
            "expected_doc": "QC-SOP-014",
            "expected_rev": "04",
            "expected_page": 2,
            "must_contain": [["90.000"], ["+/- 0.050", "±0.050", "+/-0.050"]],
            "must_not_contain": ["52.000", "+0.030"],
            "expected_status": ["answered"]
        },
        {
            "category": "OFFICIAL_SAMPLE",
            "id": "BIZ-05",
            "name": "Spindle Runout Check Schedule & Role",
            "question": "How often should spindle runout be checked, and who does it?",
            "allow_superseded": False,
            "expected_doc": "MNT-MAN-CNC-02",
            "expected_rev": "02",
            "expected_page": 1,
            "must_contain": [["monthly"], ["maintenance technician", "technician"]],
            "must_not_contain": ["operator", "daily", "annual"],
            "expected_status": ["answered"]
        },

        # --- 2. ADVERSARIAL: AL-106 UNDOCUMENTED ALARM ---
        {
            "category": "ADVERSARIAL_ALARM",
            "id": "ADV-01",
            "name": "Direct AL-106 Undocumented Query",
            "question": "What is the meaning and action for alarm AL-106 on the TX-200 lathe?",
            "allow_superseded": False,
            "expected_doc": None,
            "expected_rev": None,
            "expected_page": None,
            "must_contain": [["not documented", "unrecognized", "not found"]],
            "must_not_contain": ["spindle fault", "overheat", "way lube"],
            "expected_status": ["not_documented"]
        },
        {
            "category": "ADVERSARIAL_ALARM",
            "id": "ADV-02",
            "name": "Hallucination Bait for AL-106",
            "question": "The TX-200 manual describes AL-101 (hydraulic) and AL-104 (lube). Extrapolating from this sequence, what is alarm AL-106?",
            "allow_superseded": False,
            "expected_doc": None,
            "expected_rev": None,
            "expected_page": None,
            "must_contain": [["not documented", "unrecognized", "not found"]],
            "must_not_contain": ["coolant pressure", "hydraulic motor"],
            "expected_status": ["not_documented"]
        },

        # --- 3. ADVERSARIAL: LOTO OWNER-ABSENT EXCEPTION & SUPERVISOR BYPASS ---
        {
            "category": "ADVERSARIAL_LOTO",
            "id": "ADV-03",
            "name": "Supervisor Independent Lock Removal Attempt",
            "question": "Can a Shift Supervisor independently cut off or remove an absent technician's lockout lock if production is stopped?",
            "allow_superseded": False,
            "expected_doc": "SAF-SOP-003",
            "expected_rev": "05",
            "expected_page": 1,
            "must_contain": [["cannot", "no", "not allowed"]],
            "must_not_contain": ["yes, the supervisor can", "supervisor is authorized alone"],
            "expected_status": ["answered"]
        },
        {
            "category": "ADVERSARIAL_LOTO",
            "id": "ADV-04",
            "name": "Emergency Lock Removal Form & Verification",
            "question": "What specific form and procedural steps are mandated if a lockout lock owner cannot be reached by telephone?",
            "allow_superseded": False,
            "expected_doc": "SAF-SOP-003",
            "expected_rev": "05",
            "expected_page": 2,
            "must_contain": [["sf-12", "sf 12"], ["production manager", "safety officer"]],
            "must_not_contain": ["qf-07", "form 104"],
            "expected_status": ["answered"]
        },

        # --- 4. ADVERSARIAL: REV 03 VS REV 04 CONFLICT & GOVERNANCE ---
        {
            "category": "ADVERSARIAL_GOVERNANCE",
            "id": "ADV-05",
            "name": "Operational Sampling Plan (200 pieces)",
            "question": "What is the incoming material inspection sample size for a shipment of 200 pieces of raw material?",
            "allow_superseded": False,
            "expected_doc": "QC-SOP-014",
            "expected_rev": "04",
            "expected_page": 1,
            "must_contain": [["13", "13 pieces", "13 samples"]],
            "must_not_contain": ["5 samples for all lots", "rev 03"],
            "expected_status": ["answered"]
        },
        {
            "category": "ADVERSARIAL_GOVERNANCE",
            "id": "ADV-06",
            "name": "Direct Historical Query for Rev 03",
            "question": "What did the superseded Revision 03 of QC-SOP-014 specify regarding sample size, inspection form, and record retention?",
            "allow_superseded": True,
            "expected_doc": "QC-SOP-014",
            "expected_rev": "03",
            "expected_page": 1,
            "must_contain": [["5 samples", "5 pieces"], ["qf-07", "qf 07"], ["2 years", "two years"]],
            "must_not_contain": ["qf-07a", "5 years"],
            "expected_status": ["answered"]
        },
        {
            "category": "ADVERSARIAL_GOVERNANCE",
            "id": "ADV-07",
            "name": "Direct Conflict Resolution: 5 vs 13 samples and Quarantine Location",
            "question": "Our old shop notes say sample 5 parts and quarantine rejected lots in the Red Bin near Gate 2, but the new SOP mentions 13 parts and Cage Q1. Why is there a conflict, and which one is active?",
            "allow_superseded": True,
            "expected_doc": "QC-SOP-014",
            "expected_rev": "04",
            "expected_page": 1,
            "must_contain": [["supersed"], ["rev 04", "revision 04"]],
            "must_not_contain": [],
            "expected_status": ["answered", "conflict"]
        },

        # --- 5. ADVERSARIAL: UNSUPPORTED SPECIFICATIONS & MACHINE SCOPE ---
        {
            "category": "ADVERSARIAL_SPECS",
            "id": "ADV-08",
            "name": "Hydraulic Hose Pressure Rating (Omitted in Manual)",
            "question": "What is the exact working pressure rating or bar rating specified for the replacement hydraulic hose on the TX-200?",
            "allow_superseded": False,
            "expected_doc": "MNT-MAN-CNC-02",
            "expected_rev": "02",
            "expected_page": 2,
            "must_contain": [["not documented", "not specified", "same rating"]],
            "must_not_contain": ["250 bar", "300 bar", "3000 psi", "350 bar"],
            "expected_status": ["not_documented", "partial", "answered"]
        },
        {
            "category": "ADVERSARIAL_SPECS",
            "id": "ADV-09",
            "name": "Machine Scope (CNC-02 Coverage Boundary)",
            "question": "Does the TX-200 maintenance manual MNT-MAN-CNC-02 apply to machine CNC-02?",
            "allow_superseded": False,
            "expected_doc": "MNT-MAN-CNC-02",
            "expected_rev": "02",
            "expected_page": 1,
            "must_contain": [["no", "not apply", "does not cover"], ["cnc-04 to cnc-09", "cnc-04", "older models"]],
            "must_not_contain": ["yes, it applies to cnc-02"],
            "expected_status": ["answered"]
        },
        {
            "category": "ADVERSARIAL_SPECS",
            "id": "ADV-10",
            "name": "BH-220 Asymmetric Bore Tolerance & Method",
            "question": "What is the specified bore diameter, asymmetric tolerance, and required measurement method for the BH-220 bearing housing?",
            "allow_superseded": False,
            "expected_doc": "QC-SOP-014",
            "expected_rev": "04",
            "expected_page": 2,
            "must_contain": [["52.000"], ["+0.030", "-0.000"], ["cmm"]],
            "must_not_contain": ["90.000", "caliper"],
            "expected_status": ["answered"]
        },

        # --- 6. ADVERSARIAL: CROSS-DOCUMENT SAFETY REFERENCES ---
        {
            "category": "ADVERSARIAL_CROSS_DOC",
            "id": "ADV-11",
            "name": "Hydraulic Hose Replacement Cross-Ref to LOTO",
            "question": "What are the required safety precautions and procedure before replacing a hydraulic hose on the TX-200?",
            "allow_superseded": False,
            "expected_doc": "MNT-MAN-CNC-02",
            "expected_rev": "02",
            "expected_page": 2,
            "must_contain": [["saf-sop-003", "lockout", "loto"], ["zero", "depressurize", "drip tray"]],
            "must_not_contain": [],
            "expected_status": ["answered"]
        }
    ]

    total_tests = len(test_cases)
    passed_tests = 0
    detailed_reports = []

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/{total_tests}] [{tc['category']}] {tc['id']}: {tc['name']}", flush=True)
        print(f"      Question: {tc['question']}", flush=True)
        
        req = AskRequest(
            question=tc["question"],
            allow_superseded=tc["allow_superseded"]
        )
        
        t0 = time.perf_counter()
        resp = engine.ask(req)
        duration = round(time.perf_counter() - t0, 3)
        
        # --- Deep Inspection ---
        # 1. Status Check
        status_ok = resp.status.value in tc["expected_status"]
        
        # 2. Document & Revision Check
        retrieved_ids = resp.query_metadata.get("retrieved_chunk_ids", [])
        cited_cids = [c.chunk_id for c in resp.citations]
        
        doc_ok = True
        rev_ok = True
        page_ok = True
        
        if tc["expected_doc"]:
            cited_docs = [c.document_number for c in resp.citations if c.document_number]
            doc_ok = tc["expected_doc"] in cited_docs or any(tc["expected_doc"] in cid for cid in retrieved_ids)
            
        if tc["expected_rev"]:
            cited_revs = [c.revision for c in resp.citations if c.revision]
            rev_ok = tc["expected_rev"] in cited_revs or any(f"_R{tc['expected_rev']}_" in cid for cid in retrieved_ids)
            
        if tc["expected_page"]:
            cited_pages = [c.page_number for c in resp.citations if c.page_number]
            # Page match in citations or top retrieved chunk
            page_ok = tc["expected_page"] in cited_pages or any(f"_P0{tc['expected_page']}" in cid or f"_P{tc['expected_page']}" in cid for cid in retrieved_ids[:2])

        # 3. Grounding / Keyword Checks
        ans_lower = resp.answer.lower()
        must_pass = True
        missing_keywords = []
        for req in tc["must_contain"]:
            if isinstance(req, str):
                synonyms = [req]
            else:
                synonyms = list(req)
            if not any(syn.lower() in ans_lower for syn in synonyms):
                missing_keywords.append(synonyms[0])
        
        forbidden_keywords = []
        for kw in tc["must_not_contain"]:
            if kw.lower() in ans_lower:
                forbidden_keywords.append(kw)
                
        kw_ok = (len(missing_keywords) == 0 and len(forbidden_keywords) == 0)
        
        # 4. Citation Verification Inspection
        verified_citations = [c for c in resp.citations if c.verified]
        unverified_citations = [c for c in resp.citations if not c.verified]
        
        # Overall Verdict
        test_pass = status_ok and doc_ok and rev_ok and kw_ok
        if test_pass:
            passed_tests += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"
            
        # Display Diagnostics
        print(f"      Latency:     {duration}s ({resp.query_metadata.get('latency_ms', 0)} ms)", flush=True)
        print(f"      Status:      {resp.status.value.upper()} (Expected: {tc['expected_status']}) -> {'[OK]' if status_ok else '[FAIL]'}", flush=True)
        print(f"      Doc/Rev/Pg:  Doc: {tc['expected_doc']}, Rev: {tc['expected_rev']}, Page: {tc['expected_page']} -> {'[OK]' if (doc_ok and rev_ok and page_ok) else '[FAIL]'}", flush=True)
        print(f"      Retrieved:   {retrieved_ids}", flush=True)
        print(f"      Citations:   {len(resp.citations)} total ({len(verified_citations)} verified, {len(unverified_citations)} unverified)", flush=True)
        for c in resp.citations[:2]:
            print(f"         * [{c.chunk_id}] Doc: {c.document_number} Rev {c.revision} P.{c.page_number} (Verified: {c.verified})", flush=True)
        if missing_keywords:
            print(f"      MISSING KW:  {missing_keywords}", flush=True)
        if forbidden_keywords:
            print(f"      FORBIDDEN KW: {forbidden_keywords}", flush=True)
        print(f"      Answer:      {resp.answer[:160]}...", flush=True)
        print(f"      VERDICT:     [{verdict}]", flush=True)
        
        detailed_reports.append({
            "id": tc["id"],
            "name": tc["name"],
            "category": tc["category"],
            "question": tc["question"],
            "verdict": verdict,
            "status": resp.status.value,
            "answer": resp.answer,
            "latency_ms": resp.query_metadata.get("latency_ms", 0),
            "citations": [c.model_dump() for c in resp.citations],
            "retrieved_chunks": retrieved_ids,
            "missing_keywords": missing_keywords,
            "forbidden_keywords": forbidden_keywords,
            "status_ok": status_ok,
            "doc_ok": doc_ok,
            "rev_ok": rev_ok,
            "page_ok": page_ok,
            "kw_ok": kw_ok
        })

    accuracy = round((passed_tests / total_tests) * 100, 1)
    print("\n" + "=" * 90)
    print(f"ADVERSARIAL VALIDATION SUMMARY: {passed_tests}/{total_tests} PASSED ({accuracy}% COMPLIANCE)")
    print("=" * 90)
    
    with open("data/adversarial_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "accuracy_pct": accuracy,
            "details": detailed_reports
        }, f, indent=2)
    print("Detailed report persisted to data/adversarial_results.json")
    
    return passed_tests, total_tests

if __name__ == "__main__":
    passed, total = run_adversarial_validation()
    sys.exit(0 if passed == total else 1)

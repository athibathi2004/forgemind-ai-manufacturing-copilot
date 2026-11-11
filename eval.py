import os
import sys
import json
import time
from typing import List, Dict, Any

from forgemind.pipeline.rag_engine import RAGEngine
from forgemind.core.models import AskRequest, ResponseStatus

# 13 Required Evaluation Questions
BENCHMARK_CASES = [
    {
        "id": "Q01",
        "question": "What is the recommended coolant concentration for the TX-200 CNC turning centre?",
        "expected_status": ["answered"],
        "expected_doc": "MNT-MAN-CNC-02",
        "expected_rev": "02",
        "expected_page": 1,
        "keywords": ["6 - 8 %", "6 to 8%"],
        "allow_superseded": False,
        "notes": "Biztel Q1: Table 3 parameter row"
    },
    {
        "id": "Q02",
        "question": "The machine is showing alarm AL-104. What does it mean and what should the operator do?",
        "expected_status": ["answered"],
        "expected_doc": "MNT-MAN-CNC-02",
        "expected_rev": "02",
        "expected_page": 2,
        "keywords": ["way lube", "ISO VG 68", "reset"],
        "allow_superseded": False,
        "notes": "Biztel Q2: Common Alarms table"
    },
    {
        "id": "Q03",
        "question": "Who is allowed to remove a lockout lock from a machine?",
        "expected_status": ["answered"],
        "expected_doc": "SAF-SOP-003",
        "expected_rev": "05",
        "expected_page": 1,
        "keywords": ["own lock", "owner", "production manager", "safety officer"],
        "allow_superseded": False,
        "notes": "Biztel Q3: Must include both general rule and owner-absent exception"
    },
    {
        "id": "Q04",
        "question": "What is the outer diameter tolerance for the BH-220 bearing housing?",
        "expected_status": ["answered"],
        "expected_doc": "QC-SOP-014",
        "expected_rev": "04",
        "expected_page": 2,
        "keywords": ["90.000", "+/- 0.050"],
        "allow_superseded": False,
        "notes": "Biztel Q4: Section 6 inspection table (Rev 04)"
    },
    {
        "id": "Q05",
        "question": "How often should spindle runout be checked, and who does it?",
        "expected_status": ["answered"],
        "expected_doc": "MNT-MAN-CNC-02",
        "expected_rev": "02",
        "expected_page": 1,
        "keywords": ["monthly", "maintenance technician"],
        "allow_superseded": False,
        "notes": "Biztel Q5: Section 4 PM schedule table"
    },
    {
        "id": "Q06",
        "question": "What is the sampling requirement for a lot of 200 pieces?",
        "expected_status": ["answered"],
        "expected_doc": "QC-SOP-014",
        "expected_rev": "04",
        "expected_page": 1,
        "keywords": ["13", "samples"],
        "allow_superseded": False,
        "notes": "Domain Q6: Lot size 91-500 requires 13 samples"
    },
    {
        "id": "Q07",
        "question": "What happens if the incoming material does not have an MTC?",
        "expected_status": ["answered"],
        "expected_doc": "QC-SOP-014",
        "expected_rev": "04",
        "expected_page": 2,
        "keywords": ["hold", "mandatory"],
        "allow_superseded": False,
        "notes": "Domain Q7: Section 5 step 2"
    },
    {
        "id": "Q08",
        "question": "What is the procedure for replacing a hydraulic hose?",
        "expected_status": ["answered"],
        "expected_doc": "MNT-MAN-CNC-02",
        "expected_rev": "02",
        "expected_page": 2,
        "keywords": ["SAF-SOP-003", "zero", "drip tray"],
        "allow_superseded": False,
        "notes": "Domain Q8: Cross-references SAF-SOP-003"
    },
    {
        "id": "Q09",
        "question": "What is the pressure rating of the replacement hydraulic hose?",
        "expected_status": ["not_documented", "partial"],
        "expected_doc": "MNT-MAN-CNC-02",
        "expected_rev": "02",
        "expected_page": 2,
        "keywords": ["not documented", "not specified", "same rating"],
        "allow_superseded": False,
        "notes": "Domain Q9: Hose rating is not documented; manual only says 'same rating'"
    },
    {
        "id": "Q10",
        "question": "What is the meaning of alarm AL-106?",
        "expected_status": ["not_documented"],
        "expected_doc": None,
        "expected_rev": None,
        "expected_page": None,
        "keywords": ["not documented", "unrecognized"],
        "allow_superseded": False,
        "notes": "Domain Q10: Alarm AL-106 is absent from manual"
    },
    {
        "id": "Q11",
        "question": "What did the superseded Revision 03 specify about sampling?",
        "expected_status": ["answered"],
        "expected_doc": "QC-SOP-014",
        "expected_rev": "03",
        "expected_page": 1,
        "keywords": ["5 samples", "irrespective of lot size"],
        "allow_superseded": True,
        "notes": "Domain Q11: Superseded Rev 03 sampling rule"
    },
    {
        "id": "Q12",
        "question": "Can a Shift Supervisor independently remove an absent employee's lock?",
        "expected_status": ["answered"],
        "expected_doc": "SAF-SOP-003",
        "expected_rev": "05",
        "expected_page": 1,
        "keywords": ["no", "cannot", "both", "production manager", "safety officer"],
        "allow_superseded": False,
        "notes": "Domain Q12: Requires dual authorization"
    },
    {
        "id": "Q13",
        "question": "Does the TX-200 manual cover CNC-02?",
        "expected_status": ["answered"],
        "expected_doc": "MNT-MAN-CNC-02",
        "expected_rev": "02",
        "expected_page": 1,
        "keywords": ["no", "cnc-04 to cnc-09", "not cover"],
        "allow_superseded": False,
        "notes": "Domain Q13: Section 1 machine scope"
    }
]

def run_evaluation():
    print("=" * 80)
    print("        FORGEMIND AI — COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 80)
    
    engine = RAGEngine()
    results = []
    passed_count = 0
    
    for case in BENCHMARK_CASES:
        qid = case["id"]
        q = case["question"]
        print(f"\n[{qid}] Evaluating: {q}")
        
        req = AskRequest(
            question=q,
            allow_superseded=case["allow_superseded"]
        )
        
        t0 = time.perf_counter()
        resp = engine.ask(req)
        duration = round(time.perf_counter() - t0, 2)
        
        # Verification criteria
        status_ok = resp.status.value in case["expected_status"]
        
        # Check doc/rev/page citations if expected
        doc_ok = True
        rev_ok = True
        page_ok = True
        
        if case["expected_doc"]:
            cited_docs = [c.document_number for c in resp.citations if c.document_number]
            doc_ok = case["expected_doc"] in cited_docs or any(case["expected_doc"] in cid for cid in resp.query_metadata.get("retrieved_chunk_ids", []))
            
        if case["expected_rev"]:
            cited_revs = [c.revision for c in resp.citations if c.revision]
            rev_ok = case["expected_rev"] in cited_revs or any(f"_R{case['expected_rev']}_" in cid for cid in resp.query_metadata.get("retrieved_chunk_ids", []))

        # Check keyword presence in answer
        ans_lower = resp.answer.lower()
        kw_ok = any(kw.lower() in ans_lower for kw in case["keywords"])
        
        is_pass = status_ok and doc_ok and rev_ok and kw_ok
        if is_pass:
            passed_count += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"
            
        print(f"      Status:   {resp.status.value.upper()} (Expected: {case['expected_status']}) -> {'OK' if status_ok else 'MISMATCH'}")
        print(f"      Doc/Rev:  {case.get('expected_doc')} Rev {case.get('expected_rev')} -> {'OK' if (doc_ok and rev_ok) else 'MISMATCH'}")
        print(f"      Content:  {'OK' if kw_ok else 'KEYWORD MISSING'} (Expected keywords: {case['keywords']})")
        print(f"      Latency:  {duration}s ({resp.query_metadata.get('latency_ms', 0)}ms)")
        print(f"      Answer:   {resp.answer[:140]}...")
        print(f"      Verdict:  [{verdict}]")
        
        results.append({
            "id": qid,
            "question": q,
            "verdict": verdict,
            "actual_status": resp.status.value,
            "answer": resp.answer,
            "citations": [c.model_dump() for c in resp.citations],
            "latency_ms": resp.query_metadata.get("latency_ms", 0),
            "status_ok": status_ok,
            "doc_ok": doc_ok,
            "rev_ok": rev_ok,
            "content_ok": kw_ok,
            "notes": case["notes"]
        })

    total = len(BENCHMARK_CASES)
    acc = round((passed_count / total) * 100, 1)
    
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY: {passed_count}/{total} PASSED ({acc}% ACCURACY)")
    print("=" * 80)
    
    # Save results
    os.makedirs("data", exist_ok=True)
    out_file = os.path.join("data", "eval_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": total,
            "passed_questions": passed_count,
            "accuracy_pct": acc,
            "results": results
        }, f, indent=2)
    print(f"Results persisted to {out_file}")

if __name__ == "__main__":
    run_evaluation()

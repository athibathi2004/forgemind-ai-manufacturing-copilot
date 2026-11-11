document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const askForm = document.getElementById("ask-form");
  const questionInput = document.getElementById("question-input");
  const allowSupersededToggle = document.getElementById("allow-superseded-toggle");
  const clearBtn = document.getElementById("clear-btn");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");

  const sampleChipsContainer = document.getElementById("sample-chips-container");
  const resultsCard = document.getElementById("results-card");
  const responseStatusBadge = document.getElementById("response-status-badge");
  const queryLatency = document.getElementById("query-latency");
  const verifiedIndicator = document.getElementById("verified-indicator");
  const answerContent = document.getElementById("answer-content");

  const warningsContainer = document.getElementById("warnings-container");
  const conflictsBox = document.getElementById("conflicts-box");
  const conflictsList = document.getElementById("conflicts-list");
  const missingBox = document.getElementById("missing-box");
  const missingList = document.getElementById("missing-list");

  const evidenceToggle = document.getElementById("evidence-toggle");
  const evidenceCardsContainer = document.getElementById("evidence-cards-container");
  const evidenceCount = document.getElementById("evidence-count");
  const evidenceToggleIcon = document.getElementById("evidence-toggle-icon");

  const lanIpText = document.getElementById("lan-ip-text");
  const docCountBadge = document.getElementById("doc-count-badge");
  const engineModelBadge = document.getElementById("engine-model-badge");

  const docStatusTrigger = document.getElementById("doc-status-trigger");
  const docModal = document.getElementById("doc-modal");
  const closeModalBtn = document.getElementById("close-modal-btn");
  const qmsTableBody = document.getElementById("qms-table-body");

  let qmsDocuments = [];

  // 1. Load System Status and Document Registry
  async function loadSystemStatus() {
    try {
      const res = await fetch("/api/system/status");
      if (!res.ok) throw new Error("Failed to load status");
      const data = await res.json();

      qmsDocuments = data.documents || [];
      docCountBadge.textContent = `${data.documents_count} Docs (${data.chunks_count} Chunks)`;
      engineModelBadge.textContent = data.active_provider;
      
      const port = window.location.port ? `:${window.location.port}` : "";
      const displayIp = data.lan_ip !== "127.0.0.1" ? data.lan_ip : window.location.hostname;
      lanIpText.textContent = `LAN: ${displayIp}${port}`;

      renderQmsTable(qmsDocuments);
    } catch (err) {
      console.error("Status load error:", err);
      docCountBadge.textContent = "Error loading docs";
      engineModelBadge.textContent = "Unavailable";
    }
  }

  // 2. Load Sample Questions
  async function loadSampleQuestions() {
    try {
      const res = await fetch("/api/sample-questions");
      if (!res.ok) throw new Error("Failed to load sample questions");
      const data = await res.json();

      sampleChipsContainer.innerHTML = "";

      // Biztel official questions
      (data.biztel_questions || []).forEach((q) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "sample-chip biztel-chip";
        chip.textContent = q;
        chip.title = "BiztelAI Core Evaluation Question";
        chip.addEventListener("click", () => {
          questionInput.value = q;
          runQuery(q);
        });
        sampleChipsContainer.appendChild(chip);
      });

      // Domain verification questions
      (data.domain_verification_questions || []).forEach((q) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "sample-chip";
        chip.textContent = q;
        chip.title = "Domain Verification Question";
        chip.addEventListener("click", () => {
          questionInput.value = q;
          if (q.toLowerCase().includes("superseded") || q.toLowerCase().includes("rev 03")) {
            allowSupersededToggle.checked = true;
          }
          runQuery(q);
        });
        sampleChipsContainer.appendChild(chip);
      });
    } catch (err) {
      console.error("Sample questions error:", err);
    }
  }

  // 3. Render QMS Modal Table
  function renderQmsTable(docs) {
    qmsTableBody.innerHTML = "";
    docs.forEach((doc) => {
      const tr = document.createElement("tr");
      const statusClass = doc.document_status === "current" ? "badge-current" : "badge-superseded";
      const statusLabel = doc.document_status.toUpperCase();
      const supersedesNote = doc.superseded_revision ? ` (Supersedes Rev ${doc.superseded_revision})` : "";

      tr.innerHTML = `
        <td><strong>${doc.document_number}</strong></td>
        <td>${doc.title}</td>
        <td><span class="doc-rev-badge ${statusClass}">Rev ${doc.revision}</span></td>
        <td>${doc.effective_date}</td>
        <td><span class="doc-rev-badge ${statusClass}">${statusLabel}${supersedesNote}</span></td>
        <td>${doc.page_count}</td>
      `;
      qmsTableBody.appendChild(tr);
    });
  }

  // 4. Run Copilot Query
  async function runQuery(question) {
    if (!question || !question.trim()) return;

    // Loading State
    btnText.textContent = "Analyzing...";
    btnSpinner.classList.remove("hidden");
    submitBtn.disabled = true;

    try {
      const payload = {
        question: question.trim(),
        allow_superseded: allowSupersededToggle.checked
      };

      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Query failed to execute.");
      }

      const data = await res.json();
      displayResult(data);
    } catch (err) {
      displayErrorResult(err.message);
    } finally {
      btnText.textContent = "Search Copilot";
      btnSpinner.classList.add("hidden");
      submitBtn.disabled = false;
    }
  }

  // 5. Display Result
  function displayResult(data) {
    resultsCard.classList.remove("hidden");

    // Status Badge
    const status = (data.status || "answered").toLowerCase();
    responseStatusBadge.textContent = status.toUpperCase().replace("_", " ");
    responseStatusBadge.className = `response-status-badge status-${status}`;

    // Latency
    const latency = data.query_metadata?.latency_ms || 0;
    queryLatency.textContent = `Latency: ${latency} ms`;

    // Evidence verified indicator
    const hasVerified = data.query_metadata?.has_verified_citation !== false;
    if (hasVerified && status === "answered") {
      verifiedIndicator.textContent = "✓ Evidence Verified";
      verifiedIndicator.classList.remove("hidden");
    } else {
      verifiedIndicator.classList.add("hidden");
    }

    // Answer text
    answerContent.textContent = data.answer || "No response provided.";

    // Warnings (conflicts / missing info)
    let hasWarnings = false;
    if (data.conflicts && data.conflicts.length > 0) {
      conflictsList.innerHTML = data.conflicts.map((c) => `<li>${c}</li>`).join("");
      conflictsBox.classList.remove("hidden");
      hasWarnings = true;
    } else {
      conflictsBox.classList.add("hidden");
    }

    if (data.missing_info && data.missing_info.length > 0) {
      missingList.innerHTML = data.missing_info.map((m) => `<li>${m}</li>`).join("");
      missingBox.classList.remove("hidden");
      hasWarnings = true;
    } else {
      missingBox.classList.add("hidden");
    }

    if (hasWarnings) {
      warningsContainer.classList.remove("hidden");
    } else {
      warningsContainer.classList.add("hidden");
    }

    // Evidence Cards
    renderEvidenceCards(data.citations || []);

    // Scroll smoothly to results
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // 6. Display Error Result
  function displayErrorResult(errorMessage) {
    resultsCard.classList.remove("hidden");
    responseStatusBadge.textContent = "ERROR";
    responseStatusBadge.className = "response-status-badge status-error";
    queryLatency.textContent = "Failed";
    verifiedIndicator.classList.add("hidden");
    answerContent.textContent = `System Error: ${errorMessage}`;
    warningsContainer.classList.add("hidden");
    renderEvidenceCards([]);
  }

  // 7. Render Evidence Cards
  function renderEvidenceCards(citations) {
    evidenceCount.textContent = citations.length;
    evidenceCardsContainer.innerHTML = "";

    if (citations.length === 0) {
      evidenceCardsContainer.innerHTML = `
        <div style="font-size: 0.85rem; color: var(--text-dim); padding: 0.5rem 0;">
          No direct supporting passages cited for this query.
        </div>
      `;
      return;
    }

    citations.forEach((cit) => {
      const card = document.createElement("div");
      const isSuperseded = cit.revision === "03";
      card.className = `evidence-card ${isSuperseded ? "superseded-card" : ""}`;

      const statusBadge = isSuperseded
        ? `<span class="doc-rev-badge badge-superseded">Rev ${cit.revision} [SUPERSEDED]</span>`
        : `<span class="doc-rev-badge badge-current">Rev ${cit.revision || "Latest"} [CURRENT]</span>`;

      const pageBadge = cit.page_number ? `<span class="doc-page-badge">Page ${cit.page_number}</span>` : "";
      const sectionBadge = cit.section_title ? `<span class="doc-page-badge">${cit.section_title}</span>` : "";

      card.innerHTML = `
        <div class="evidence-card-header">
          <div class="evidence-doc-info">
            <span class="doc-number">${cit.document_number || cit.chunk_id}</span>
            ${statusBadge}
            ${pageBadge}
            ${sectionBadge}
          </div>
          <div class="meta-indicators">
            ${cit.verified ? '<span class="verified-tag">✓ Verified Source Passage</span>' : ''}
          </div>
        </div>
        <div class="evidence-passage">
          ${cit.source_passage || cit.quote}
        </div>
      `;
      evidenceCardsContainer.appendChild(card);
    });
  }

  // Event Listeners
  askForm.addEventListener("submit", (e) => {
    e.preventDefault();
    runQuery(questionInput.value);
  });

  clearBtn.addEventListener("click", () => {
    questionInput.value = "";
    resultsCard.classList.add("hidden");
    questionInput.focus();
  });

  evidenceToggle.addEventListener("click", () => {
    evidenceCardsContainer.classList.toggle("hidden");
    evidenceToggleIcon.textContent = evidenceCardsContainer.classList.contains("hidden") ? "▶" : "▼";
  });

  docStatusTrigger.addEventListener("click", () => {
    docModal.classList.remove("hidden");
  });

  closeModalBtn.addEventListener("click", () => {
    docModal.classList.add("hidden");
  });

  window.addEventListener("click", (e) => {
    if (e.target === docModal) {
      docModal.classList.add("hidden");
    }
  });

  // Init
  loadSystemStatus();
  loadSampleQuestions();
});

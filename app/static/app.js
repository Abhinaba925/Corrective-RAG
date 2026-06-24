const statusEl = document.querySelector("#status");
const indexNameEl = document.querySelector("#indexName");
const ingestForm = document.querySelector("#ingestForm");
const queryForm = document.querySelector("#queryForm");
const ingestButton = document.querySelector("#ingestButton");
const askButton = document.querySelector("#askButton");
const ingestResult = document.querySelector("#ingestResult");
const answerEl = document.querySelector("#answer");
const sourcesEl = document.querySelector("#sources");
const metaEl = document.querySelector("#meta");

function setStatus(text, className = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${className}`.trim();
}

async function parseJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }
  return data;
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await parseJson(response);
    indexNameEl.textContent = data.index_name;
    if (data.configured) {
      setStatus("Configured", "ready");
    } else {
      setStatus(`Missing ${data.missing_env.join(", ")}`, "missing");
    }
  } catch (error) {
    setStatus("Health check failed", "missing");
  }
}

function setBusy(button, busy, text) {
  button.disabled = busy;
  button.textContent = busy ? "Working" : text;
}

ingestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  ingestResult.className = "result";
  ingestResult.textContent = "";
  setBusy(ingestButton, true, "Ingest PDF");

  try {
    const formData = new FormData();
    const file = document.querySelector("#pdfFile").files[0];
    formData.append("file", file);
    formData.append("namespace", document.querySelector("#ingestNamespace").value);
    formData.append("chunk_size", document.querySelector("#chunkSize").value);
    formData.append("chunk_overlap", document.querySelector("#chunkOverlap").value);

    const response = await fetch("/api/ingest", {
      method: "POST",
      body: formData,
    });
    const data = await parseJson(response);
    ingestResult.textContent = `${data.chunks} chunks from ${data.pages} pages indexed in ${data.index_name}.`;
  } catch (error) {
    ingestResult.className = "result error";
    ingestResult.textContent = error.message;
  } finally {
    setBusy(ingestButton, false, "Ingest PDF");
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  answerEl.textContent = "";
  sourcesEl.innerHTML = "";
  metaEl.textContent = "";
  setBusy(askButton, true, "Ask");

  try {
    const payload = {
      query: document.querySelector("#question").value,
      mode: document.querySelector("#mode").value,
      namespace: document.querySelector("#queryNamespace").value,
    };
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await parseJson(response);
    answerEl.textContent = data.answer || "No answer returned.";
    metaEl.textContent =
      data.mode === "advanced"
        ? `Advanced CRAG - retries ${data.retries} - ${data.rewritten_query || ""}`
        : "Standard RAG";

    if (data.sources.length) {
      const fragment = document.createDocumentFragment();
      data.sources.forEach((source, index) => {
        const item = document.createElement("article");
        item.className = "source";

        const title = document.createElement("h3");
        const page = source.page ? `Page ${source.page}` : "Page ?";
        const score = source.score === null ? "" : ` - score ${source.score}`;
        title.textContent = `Source ${index + 1} - ${page}${score}`;

        const preview = document.createElement("p");
        preview.textContent = source.preview;

        item.append(title, preview);
        fragment.append(item);
      });
      sourcesEl.append(fragment);
    }
  } catch (error) {
    answerEl.textContent = error.message;
  } finally {
    setBusy(askButton, false, "Ask");
  }
});

loadHealth();

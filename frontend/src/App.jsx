import { useEffect, useMemo, useState } from "react";

const DEFAULT_BACKEND = "http://127.0.0.1:8000";

function Card({ title, subtitle, children }) {
  return (
    <section className="card">
      <div className="card__header">
        <h2>{title}</h2>
        {subtitle && <p className="card__subtitle">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}

function Status({ tone = "neutral", children }) {
  return <div className={`status status--${tone}`}>{children}</div>;
}

export default function App() {
  const backendBase = useMemo(
    () => import.meta.env.VITE_BACKEND_URL || DEFAULT_BACKEND,
    [],
  );

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [retrieved, setRetrieved] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    loadTransactions();
  }, []);

  const handleAsk = async () => {
    setError("");
    setAnswer("");
    setRetrieved("");
    if (!question.trim()) {
      setError("Please enter a question first.");
      return;
    }
    setLoading(true);
    try {
      const resp = await fetch(`${backendBase}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });
      if (!resp.ok) {
        throw new Error(`Backend error: ${resp.status}`);
      }
      const data = await resp.json();
      setAnswer(data.answer);
      setRetrieved(data.retrieved_context || "");
    } catch (e) {
      console.error(e);
      setError(e.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
  };

  const handleUpload = async () => {
    setUploadStatus("");
    setError("");
    if (!file) {
      setError("Please choose an image of a receipt first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);
    try {
      const resp = await fetch(`${backendBase}/upload_receipt`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to upload receipt");
      }
      const data = await resp.json();
      setUploadStatus(
        "Receipt processed! Parsed transaction: " +
          JSON.stringify(data.parsed_transaction, null, 2),
      );
      await loadTransactions();
    } catch (e) {
      console.error(e);
      setError(e.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const loadTransactions = async () => {
    try {
      const resp = await fetch(`${backendBase}/transactions`);
      if (!resp.ok) {
        throw new Error("Failed to fetch transactions");
      }
      const data = await resp.json();
      setTransactions(data);
    } catch (e) {
      console.error(e);
      setError(e.message || "Failed to fetch transactions");
    }
  };

  const handleReset = async () => {
    setError("");
    if (!window.confirm("Delete all stored expenses and vector data? This cannot be undone.")) {
      return;
    }
    setResetting(true);
    try {
      const resp = await fetch(`${backendBase}/reset_data`, { method: "POST" });
      if (!resp.ok) {
        throw new Error("Failed to reset data");
      }
      setAnswer("");
      setRetrieved("");
      setFile(null);
      setUploadStatus("");
      setQuestion("");
      await loadTransactions();
      setUploadStatus("All data cleared. Start fresh by uploading new receipts.");
    } catch (e) {
      console.error(e);
      setError(e.message || "Reset failed");
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div className="pill">Agentic RAG · OCR · LangGraph</div>
        <h1>Personal Finance Advisor</h1>
        <p>
          Upload receipt images, then ask natural language questions about your
          spending. The backend runs OCR, parses the receipt, stores it in a
          vector DB, and uses a LangGraph workflow to answer with context.
        </p>
        <div className="hero__meta">
          <span className="meta">Backend: FastAPI @ {backendBase}</span>
          <span className="meta">Frontend: Vite + React</span>
        </div>
      </header>

      <div className="grid two">
        <Card
          title="Upload a receipt"
          subtitle="We run OCR + simple parsing and add the transaction to your vector store."
        >
          <div className="stack">
            <label className="file-input">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                disabled={uploading}
              />
              <span>{file ? file.name : "Choose an image file"}</span>
            </label>
            <div className="actions">
              <button onClick={handleUpload} disabled={uploading}>
                {uploading ? "Processing..." : "Upload & Parse"}
              </button>
              <button
                className="ghost"
                onClick={() => setFile(null)}
                disabled={uploading || !file}
              >
                Clear
              </button>
            </div>
            {uploadStatus && <Status tone="success">{uploadStatus}</Status>}
          </div>
        </Card>

        <Card
          title="Ask a finance question"
          subtitle='Examples: "How much did I spend on groceries last month?" or "What are my biggest spending categories?"'
        >
          <div className="stack">
            <textarea
              rows="4"
              placeholder="Type your question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
            <div className="actions">
              <button onClick={handleAsk} disabled={loading}>
                {loading ? "Thinking..." : "Ask Advisor"}
              </button>
              <button className="ghost" onClick={() => setQuestion("")}>
                Clear
              </button>
            </div>
          </div>
        </Card>
      </div>

      {error && (
        <Status tone="error">Error: {error}</Status>
      )}

      {answer && (
        <Card title="Answer">
          <pre className="codeblock">{answer}</pre>
        </Card>
      )}

      {retrieved && (
        <Card title="Retrieved context">
          <pre className="codeblock">{retrieved}</pre>
        </Card>
      )}

      <Card
        title="Stored transactions"
        subtitle="Fetched from the backend; refresh after new uploads."
      >
        <div className="actions actions--end">
          <button className="ghost" onClick={loadTransactions}>
            Refresh
          </button>
          <button className="ghost danger" onClick={handleReset} disabled={resetting}>
            {resetting ? "Clearing..." : "Delete all data"}
          </button>
        </div>
        {transactions.length === 0 ? (
          <p className="muted">No transactions yet. Upload a receipt to get started.</p>
        ) : (
          <div className="table-wrapper">
            <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Merchant</th>
                  <th>Category</th>
                  <th>Amount</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((tx) => (
                  <tr key={tx.id}>
                    <td>{tx.date || "—"}</td>
                    <td>{tx.merchant || "—"}</td>
                    <td>{tx.category || "—"}</td>
                    <td>
                      {tx.amount != null
                        ? `${tx.amount} ${tx.currency || ""}`.trim()
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

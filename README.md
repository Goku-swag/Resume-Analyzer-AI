# AI Resume Analyzer (FastAPI + Groq)

Upload a resume (PDF/DOCX/TXT), optionally paste a job description, and get
structured AI feedback: overall score, ATS friendliness, strengths,
weaknesses, missing keywords, section-by-section feedback, and rewritten
bullet point examples.

## 1. Setup

```bash
cd resume_analyzer
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure your Groq API key

Get a free key at https://console.groq.com/keys, then:

```bash
cp .env.example .env
# edit .env and paste your key
export $(cat .env | xargs)      # macOS/Linux quick way to load it
```

Or just set it directly:

```bash
export GROQ_API_KEY="gsk_xxx..."        # macOS/Linux
set GROQ_API_KEY=gsk_xxx...             # Windows cmd
```

## 3. Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. Upload a resume and click
"Analyze Resume".

## API

- `GET /api/health` — health check, confirms whether `GROQ_API_KEY` is set.
- `POST /api/analyze` — multipart form:
  - `file`: resume file (`.pdf`, `.docx`, `.txt`, max 5MB)
  - `job_description` (optional): plain text job description

Example with curl:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@/path/to/resume.pdf" \
  -F "job_description=Senior backend engineer, Python, AWS, 5+ years"
```

## Error handling included

- Invalid/missing file, unsupported extension, oversized file → 400/413
- Unreadable/empty/scanned PDF, empty DOCX/TXT → 422
- Missing `GROQ_API_KEY` → 500 with a clear message
- Groq rate limits → 429
- Groq connection/API errors → 502
- Malformed JSON from the model → 502 (model is also forced into JSON mode
  via `response_format={"type": "json_object"}` to minimize this)
- A global exception handler prevents raw tracebacks from leaking to clients

## Notes

- Default model: `llama-3.3-70b-versatile`. Override with `GROQ_MODEL` env var.
- Resume text is truncated to ~12,000 characters before being sent to the
  model to stay within context/token limits.
- No data is persisted to disk or a database — everything is processed
  in-memory per request.

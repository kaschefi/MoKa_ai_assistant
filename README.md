# Cozmo AI Assistant

An AI-powered assistant built around the **Anki Cozmo** robot. The system uses a state-of-the-art two-layer intelligence pipeline: **Layer 1** fast semantic reflexes (50ms latency) for instant physical and laptop commands, and **Layer 2** a dynamic **LangGraph-powered AI brain** that uses local LLMs (`qwen2.5:1.5b` via Ollama) and a **Vector RAG Tool Retrieval Index** for complex natural conversation, Google Calendar management, and advanced search agents (Weather, Tavily MCP, Web Search). All features are exposed via an interactive console launcher, voice control, or a **FastAPI** REST bridge.

---

## Features

| Feature | Description |
|---|---|
| **Interactive Launcher** | Unified `main.py` console with a menu to run Terminal Mode (repl with brain) or Cozmo Mode (robot + API server) |
| **Voice Input** | Wake-word listener (`"hey buddy"`) captures mic audio, transcribes via Google Speech Recognition, and routes to Layer 1 reflexes or LangGraph |
| **Text-to-Speech** | Speeds responses using Microsoft Edge TTS, supporting both **English & Persian (Farsi)** with high-fidelity neural voices |
| **Layer 1 Semantic Router** | Instant intent matching (~50ms) using `semantic-router` + `FastEmbed` for latency-critical commands (bypasses LLM) |
| **Dynamic Layer 2 Tool RAG** | Rather than bloating LLM prompts with static definitions, tools are indexed in an in-memory vector database (**FAISS** + `BAAI/bge-small-en-v1.5` embeddings) and dynamically injected into the router's context based on relevance |
| **Tavily MCP Integration** | Executes highly optimized, real-time web searches using the official Tavily **Model Context Protocol (MCP)** server spawned via standard `npx` stdio client pipes |
| **Local Laptop Automation Setups** | Instant laptop routines for **Gaming Mode** (launches Steam, CS2, Discord), **Study Mode** (opens Moodle, Gemini, NotebookLM), and **Coding Mode** (opens PyCharm, GitHub, Gemini) |
| **Specialized Weather Agent** | A ReAct agent that queries the `wttr.in` API to provide real-time, conversational weather updates |
| **Google Calendar Integration** | Multi-step calendar manager that queries, creates, moves, or deletes appointments using an **n8n** webhook connected to Gemini API |
| **Autonomous Docking** | Visual ArUco marker navigation — Cozmo activates his camera, scans for the charger (Marker ID 0), steers toward it, and backs onto the charging pins |
| **Face Expressions** | Dynamic rendering of graphics, countdown timers, and weather info directly onto Cozmo's 128×64 OLED face display |
| **Timer** | Runs asynchronous countdown timers with real-time MM:SS face updates |
| **FastAPI REST Bridge** | Exposes all physical actions (docking, speak, timer, face expressions) as HTTP endpoints for external triggers |

---

## Architecture

```
                       User Input (Terminal / Voice)
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │         Layer 1 Semantic Router          │  ← Fast Embeddings (~50ms)
                │      (reflex_registry + FastEmbed)       │    Bypasses LLM entirely.
                └────────────────────┬─────────────────────┘
                                     │ No reflex match
                                     ▼
                ┌──────────────────────────────────────────┐
                │        Layer 2: LangGraph Brain          │
                │                                          │
                │  ┌────────────────────────────────────┐  │
                │  │    Tool Retrieval Node (FAISS)     │  │  ← Dynamic Tool RAG Selection
                │  └─────────────────┬──────────────────┘  │
                │                    │ top 2 matched tools
                │                    ▼
                │  ┌────────────────────────────────────┐  │
                │  │       Router Supervisor (LLM)      │  │  ← Structured classifier
                │  └────────┬────────┬────────┬─────────┘  │
                └───────────┼────────┼────────┼────────────┘
                            │        │        │
                            ▼        ▼        ▼
                      Calendar    Weather  Web Search / Casual Chat
                       (n8n)     (ReAct)   (Tavily MCP / Ollama)
```

### 1. Layer 1 Semantic Router (Fast Reflexes)
Embedding-based semantic lookup. Utterances are mapped to a local registry of python operations (`core/registry.py`) enabling instantaneous response for:
*   Physical actions: Autonomous charger docking.
*   System operations: Date, time, and laptop configurations (Gaming/Coding/Study setups).

### 2. Layer 2 LangGraph Brain (Dynamic Tool RAG)
For complex inputs, the system triggers a stateful graph:
1.  **Tool Retrieval Node**: Uses `FAISS` to run similarity search on the user's query against registered tool schemas, fetching only the top 2 candidates.
2.  **Route Query**: Constructs a system prompt with only the retrieved candidate tools and uses local LLM (`qwen2.5:1.5b`) to yield a structured `RouteDecision`.
3.  **Specialized Workers**: Routes to n8n (Google Calendar, Web Search), ReAct agent (Weather), or falls back to casual conversational chat (`chat_node`).

---

## Project Structure

The project has a highly modular architecture separating physical controls, digital integrations, state schemas, and core routing logic:

```
cozmo_ai_assistant/
│
├── main.py                     # Entry point launcher (Terminal Mode / Cozmo Mode menu)
├── Launch_Cozmo.bat            # Windows startup script to execute Terminal Mode
├── roadmap.md                  # Project milestones and task backlog
├── README.md                   # Full system documentation
│
├── core/
│   ├── __init__.py
│   ├── connection.py           # Singleton Cozmo hardware connection manager
│   ├── cozmo_mode.py           # FastAPI application server and REST endpoint routing
│   ├── registry.py             # Decorator class for low-latency Layer 1 reflex registration
│   ├── router.py               # LangGraph state machine flow, supervisor, and node workers
│   ├── semantic_layer.py       # Layer 1 semantic matching router & package-wide action loader
│   ├── terminal_mode.py        # Terminal REPL chat client with n8n/Ollama auto-initialization
│   └── tool_vector_db.py       # FAISS vector store bridge for dynamic tool registration & retrieval
│
├── actions/
│   ├── physical/               # Robot hardware controls
│   │   ├── __init__.py
│   │   ├── charger.py          # Vision-guided docking using OpenCV ArUco detector (Marker ID 0)
│   │   ├── face.py             # OLED canvas draw actions (Timer MM:SS, weather details, thinking indicator)
│   │   ├── listen.py           # Speech recognition wake-word parser ("hey buddy") and FastAPI/n8n forwarder
│   │   ├── speak.py            # edge-tts engine + Persian Gemma translator + 22kHz wav converter
│   │   └── timer.py            # Asynchronous countdown clock controller
│   │
│   └── digital/                # Digital APIs & Agent integrations
│       ├── __init__.py
│       ├── langchain_agents.py # Weather ReAct agent utilizing wttr.in tool & prompt engineering
│       ├── MCPs.py             # Tavily search powered by standard Model Context Protocol client via npx
│       ├── n8n_agents.py       # n8n webhook connectors for Google Calendar and web searching
│       ├── setups.py           # OS-level workstation launchers (Gaming, Study, Coding routines)
│       └── system_tools.py     # System action registry (Date and Time responses)
│
├── schemas/
│   ├── __init__.py
│   └── request_models.py       # Pydantic models for REST API requests and LangGraph TypedDict state
│
└── utils/
    ├── __init__.py
    └── logger.py               # Centralized logging configurations
```

---

## Prerequisites

### Hardware
*   **Anki Cozmo** robot + USB charger base + Android/iOS device running Cozmo app in SDK mode (required for physical Cozmo Mode).

### Software
*   Python **3.10+** (Python 3.11 recommended).
*   [**Ollama**](https://ollama.com/) running locally.
*   [**n8n**](https://n8n.io/) installed globally (`npm install -g n8n`) or running in your environment (auto-started by launcher).
*   **FFmpeg** (added to your system PATH; required by `pydub` for streaming speech audio to Cozmo).

### Node Packages
*   `tavily-mcp` (run automatically via `npx` during search).

### Python Libraries
Install the requirements using standard pip:
```bash
pip install pycozmo fastapi uvicorn langchain-ollama langchain-community langgraph semantic-router fastembed edge-tts pydub deep-translator opencv-python Pillow requests speechrecognition mcp python-dotenv faiss-cpu
```

Or sync with the project's **uv** configurations:
```bash
uv sync
```

---

## Getting Started

### 1. Environment Configuration

Create a `.env` file in the project root:
```env
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2. Fetch the Local AI Models

Pull the required Ollama models:
```bash
ollama pull qwen2.5:1.5b
```

### 3. Launch the Assistant

Run the unified launcher:
```bash
python main.py
```

*   **Option 1 (Terminal Mode)**: Launches the terminal chat interface. This will automatically scan ports, boot up n8n in the background, check Ollama's availability, compile the tool registry index, and open the command prompt.
*   **Option 2 (Cozmo Mode)**: Connects to the physical robot and opens a FastAPI REST bridge on port `8000`.

### 4. Enable Voice Wake-Word Activation (Optional)

Start the listener in a separate terminal:
```bash
python actions/physical/listen.py
```
Simply speak **"hey buddy"** followed by any command (e.g. *"hey buddy, set it for coding"* or *"hey buddy, what's today's date"*).

---

## Technical Highlights

### 1. Dynamic Tool RAG Retrieval
Instead of feeding all available tool instructions into the LLM system prompt—which decreases latency and accuracy—this application utilizes a FAISS-backed Vector Database registry:
```python
# actions/digital/n8n_agents.py
tool_rag_registry.register_tool_schema(
    name="calendar_node",
    description="Manages Google Calendar. Use this if the user wants to check, create, move, change, or delete meetings, events, appointments, or schedules."
)
```
Upon a user query, the `tool_retrieval_node` matches the query vector with the tool embeddings and passes the select matched tools to the Router LLM.

### 2. Standard Model Context Protocol (MCP) Client
Using standard `mcp` stdio client parameters, the Tavily search tool operates dynamically:
```python
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "tavily-mcp@latest"],
    env=os.environ.copy(),
    extra_spawn_args={"shell": True}
)
```
This spawns the standard Tavily MCP package, executes a query, and handles communication perfectly, bypassing bulky client frameworks.

### 3. Low-Latency Local Automation Reflexes
Workstation presets are tied to local shell utilities:
*   **Gaming**: Executes custom URI protocols (`steam://`) and queries paths to boot update launchers before executing Discord update commands.
*   **Coding & Study**: Performs multi-tab browser dispatch routines (`webbrowser.open`) and searches registry folders dynamically using glob matching to execute JetBrains IDEs.

---

## License

This project is for educational, research, and hobby purposes.
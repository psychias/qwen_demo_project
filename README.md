# NanoChat - Qwen1.5-4B Demo

A minimal chat interface for testing the Qwen1.5-4B language model locally.

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM) or CPU (slower)
- ~10GB disk space for model weights

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Engine

```bash
python engine.py
```

The first run will download the model (~8GB). This may take several minutes depending on your internet connection.

### 3. Open the Chat Interface

Once the engine is running, simply open your browser and navigate to:

```
http://localhost:8000
```

The FastAPI server automatically serves both the backend API and the frontend HTML. No additional server needed!

## 📁 Project Structure

```
qwen_demo/
├── engine.py          # FastAPI backend server
├── index.html         # Chat frontend
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## ⚙️ Configuration

### Model Settings

Edit `engine.py` to change:

```python
MODEL_ID = "Qwen/Qwen1.5-4B-Chat"  # Model to use
```

Available Qwen1.5 variants:
- `Qwen/Qwen1.5-0.5B-Chat` - Smallest, fastest
- `Qwen/Qwen1.5-1.8B-Chat` - Small
- `Qwen/Qwen1.5-4B-Chat` - Medium (default)
- `Qwen/Qwen1.5-7B-Chat` - Large
- `Qwen/Qwen1.5-14B-Chat` - Very large

### Server Settings

```bash
# Change host/port
python -c "import uvicorn; uvicorn.run('engine:app', host='0.0.0.0', port=8080)"
```

### Frontend Settings

Edit `API_URL` in `index.html` if your server runs on a different port:

```javascript
const API_URL = 'http://localhost:8000';
```

## 💬 Chat Commands

Type these in the chat input:

| Command | Description |
|---------|-------------|
| `/temperature [value]` | Get/set temperature (0.0-2.0) |
| `/topk [value]` | Get/set top-k sampling (1-200) |
| `/topp [value]` | Get/set top-p/nucleus sampling (0.0-1.0) |
| `/maxtokens [value]` | Get/set max response tokens (1-4096) |
| `/settings` | Show all current settings |
| `/clear` | Clear conversation |
| `/help` | Show help message |

## 🎯 Features

- **Streaming responses** - See tokens as they're generated
- **Message editing** - Click any user message to edit and regenerate
- **Response regeneration** - Click any assistant message to regenerate
- **Keyboard shortcuts** - Ctrl+Shift+N for new conversation
- **Status indicator** - Shows connection status (green = connected)
- **Adjustable parameters** - Temperature, top-k, top-p, max tokens

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/chat/completions` | POST | Chat completion (streaming) |

### Example API Request

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.8,
    "max_tokens": 512
  }'
```

## 🐛 Troubleshooting

### "Engine not running" error
- Make sure `engine.py` is running
- Check the terminal for errors
- Verify the model downloaded correctly

### CUDA out of memory
- Use a smaller model variant
- Reduce `max_tokens`
- Close other GPU applications

### Slow generation on CPU
- This is expected; GPU is strongly recommended
- Try a smaller model like `Qwen1.5-0.5B-Chat`

### Port already in use
- Another process is using port 8000
- Stop any running `python -m http.server` or other servers
- Or change the port in `engine.py` (line 249)

## 📝 License

MIT License - Feel free to use and modify!

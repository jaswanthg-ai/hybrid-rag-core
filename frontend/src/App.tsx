import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: { text: string; score: number }[];
}

function App() {
  const [runId, setRunId] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadInfo, setUploadInfo] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState('');
  const [expandedSrc, setExpandedSrc] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamText]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadInfo('');
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Upload failed');
      }
      const data = await res.json();
      setRunId(data.run_id);
      setUploadInfo(data.message);
      setMessages([]);
    } catch (err: any) {
      setUploadInfo(`Error: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const sendMessage = async () => {
    const msg = input.trim();
    if (!msg || !runId || streaming) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: msg }]);
    setStreaming(true);
    setStreamText('');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, message: msg }),
      });
      if (!res.ok) throw new Error(`${res.status}`);

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No body');

      const decoder = new TextDecoder();
      let full = '';
      let sources: { text: string; score: number }[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value, { stream: true }).split('\n')) {
          if (!line.startsWith('data: ')) continue;
          try {
            const d = JSON.parse(line.slice(6));
            if (d.type === 'token') { full += d.content; setStreamText(full); }
            if (d.type === 'sources') sources = d.chunks || [];
          } catch {}
        }
      }
      setMessages(prev => [...prev, { role: 'assistant', content: full, sources }]);
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setStreaming(false);
      setStreamText('');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="border-b bg-white px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <h1 className="text-lg font-bold text-gray-900">hybrid-rag-core</h1>
          <div className="flex items-center gap-3">
            {uploadInfo && <span className="text-xs text-green-600">{uploadInfo}</span>}
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {uploading ? 'Processing…' : 'Upload PDF'}
            </button>
            <input ref={fileRef} type="file" accept=".pdf" className="hidden" onChange={handleUpload} />
          </div>
        </div>
      </header>

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-6 space-y-4">
          {!runId && messages.length === 0 && (
            <div className="text-center py-20 text-gray-400">
              <p className="text-lg">Upload a PDF to start chatting</p>
              <p className="text-sm mt-2">The document will be automatically processed for Q&A</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] space-y-1`}>
                <div className={`rounded-lg px-4 py-3 text-sm whitespace-pre-wrap ${
                  msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-gray-200 text-gray-900'
                }`}>
                  {msg.content}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <button
                    onClick={() => setExpandedSrc(expandedSrc === i ? null : i)}
                    className="text-xs text-blue-500 hover:text-blue-700 px-1"
                  >
                    {expandedSrc === i ? 'Hide sources' : `${msg.sources.length} sources`}
                  </button>
                )}
                {expandedSrc === i && msg.sources?.map((s, j) => (
                  <div key={j} className="rounded bg-gray-50 border border-gray-100 p-2 text-xs text-gray-600">
                    <span className="text-gray-400">Score: {s.score}</span>
                    <p className="mt-1 whitespace-pre-wrap">{s.text}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}

          {streaming && streamText && (
            <div className="flex justify-start">
              <div className="max-w-[85%] rounded-lg px-4 py-3 text-sm bg-white border border-gray-200 text-gray-900 whitespace-pre-wrap">
                {streamText}<span className="inline-block w-1 h-4 bg-gray-400 animate-pulse ml-0.5" />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t bg-white px-6 py-4">
        <div className="max-w-3xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendMessage()}
            placeholder={runId ? 'Ask a question about your document…' : 'Upload a PDF first'}
            disabled={!runId || streaming}
            className="flex-1 rounded-lg border border-gray-300 px-4 py-3 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50 disabled:bg-gray-50"
          />
          <button
            onClick={sendMessage}
            disabled={!runId || streaming || !input.trim()}
            className="rounded-lg bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;

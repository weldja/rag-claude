import { useState, useEffect, useRef, useCallback } from 'react'

// Simple markdown renderer — handles bold, italic, bullets, numbered lists, line breaks
function Markdown({ text }) {
  if (!text) return null
  
  const lines = text.split('\n')
  const elements = []
  let i = 0
  
  while (i < lines.length) {
    const line = lines[i]
    
    // Numbered list
    if (/^\d+\.\s/.test(line)) {
      const items = []
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        items.push(renderInline(lines[i].replace(/^\d+\.\s/, '')))
        i++
      }
      elements.push(<ol key={i} className="list-decimal pl-5 my-1.5 space-y-1">{items.map((item, j) => <li key={j} className="text-sm leading-relaxed">{item}</li>)}</ol>)
      continue
    }
    
    // Bullet list
    if (/^[-*]\s/.test(line)) {
      const items = []
      while (i < lines.length && /^[-*]\s/.test(lines[i])) {
        items.push(renderInline(lines[i].replace(/^[-*]\s/, '')))
        i++
      }
      elements.push(<ul key={i} className="list-disc pl-5 my-1.5 space-y-1">{items.map((item, j) => <li key={j} className="text-sm leading-relaxed">{item}</li>)}</ul>)
      continue
    }
    
    // Empty line — spacer
    if (line.trim() === '') {
      elements.push(<div key={i} className="h-1.5" />)
      i++
      continue
    }
    
    // Normal paragraph
    elements.push(<p key={i} className="text-sm leading-relaxed">{renderInline(line)}</p>)
    i++
  }
  
  return <div className="space-y-0.5">{elements}</div>
}

function renderInline(text) {
  // Bold + italic, bold, italic, inline code
  const parts = []
  const regex = /\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`/g
  let last = 0
  let m
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index))
    if (m[1]) parts.push(<strong key={m.index}><em>{m[1]}</em></strong>)
    else if (m[2]) parts.push(<strong key={m.index} className="font-semibold text-slate-900">{m[2]}</strong>)
    else if (m[3]) parts.push(<em key={m.index}>{m[3]}</em>)
    else if (m[4]) parts.push(<code key={m.index} className="text-xs bg-slate-100 px-1 py-0.5 rounded font-mono">{m[4]}</code>)
    last = m.index + m[0].length
  }
  if (last < text.length) parts.push(text.slice(last))
  return parts.length > 0 ? parts : text
}

// ─────────────────────────────────────────────
// API helpers
// ─────────────────────────────────────────────

// Session ID — persists in localStorage across page refreshes
function getSessionId() {
  let id = localStorage.getItem('weldai_session_id')
  if (!id) {
    id = 'sess_' + Math.random().toString(36).slice(2) + Date.now().toString(36)
    localStorage.setItem('weldai_session_id', id)
  }
  return id
}

const SESSION_ID = getSessionId()

const api = {
  get: (path) => fetch(path).then(r => r.json()),
  post: (path, body) => fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(r => r.json()),
  del: (path) => fetch(path, { method: 'DELETE' }).then(r => r.json()),
}

function streamSSE(path, body, onEvent) {
  let cancelled = false
  const run = async () => {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    while (!cancelled) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop()
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try { onEvent(JSON.parse(line.slice(6))) } catch {}
        }
      }
    }
  }
  run().catch(console.error)
  return () => { cancelled = true }
}

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────

function Dots() {
  return (
    <span className="flex gap-1 items-center">
      <span className="dot bg-slate-400" />
      <span className="dot bg-slate-400" />
      <span className="dot bg-slate-400" />
    </span>
  )
}

function FileIcon({ ext }) {
  const colours = {
    pdf:  'bg-red-100 text-red-700',
    docx: 'bg-blue-100 text-blue-700',
    doc:  'bg-blue-100 text-blue-700',
    pptx: 'bg-amber-100 text-amber-700',
    ppt:  'bg-amber-100 text-amber-700',
    xlsx: 'bg-emerald-100 text-emerald-700',
    xls:  'bg-emerald-100 text-emerald-700',
    csv:  'bg-emerald-100 text-emerald-700',
    txt:  'bg-slate-100 text-slate-600',
  }
  return (
    <span className={`text-xs font-bold px-1.5 py-0.5 rounded uppercase ${colours[ext] || 'bg-slate-100 text-slate-600'}`}>
      {ext}
    </span>
  )
}

function SourceCards({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources?.length) return null
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-700 transition-colors"
      >
        <span>{open ? '▾' : '▸'}</span>
        <span>{sources.length} source{sources.length > 1 ? 's' : ''} referenced</span>
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2 fade-in">
          {sources.map((src, i) => (
            <div key={i} className="bg-slate-50 border border-slate-200 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-base">{src.icon || '📄'}</span>
                <span className="text-xs font-semibold text-slate-700 flex-1">{src.display_name}</span>
                <span className="text-xs bg-blue-50 text-blue-600 rounded-full px-2 py-0.5 font-medium">#{i + 1}</span>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed line-clamp-3">{src.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function MessageBubble({ msg, accent }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex gap-3 fade-in ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      {!isUser && (
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center text-white text-xs font-bold flex-shrink-0 mt-0.5"
          style={{ background: `linear-gradient(135deg, ${accent}, ${accent}cc)` }}
        >
          AI
        </div>
      )}
      <div className={`flex-1 ${isUser ? 'flex justify-end' : ''}`}>
        {isUser ? (
          <div
            className="inline-block px-4 py-2.5 rounded-2xl rounded-tr-sm text-white text-sm leading-relaxed max-w-md shadow-sm"
            style={{ background: accent }}
          >
            {msg.content}
          </div>
        ) : (
          <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm max-w-3xl">
            {msg.streaming ? (
              <div className="text-slate-800">
                <Markdown text={msg.content} />
                <span className="cursor" />
              </div>
            ) : msg.status ? (
              <span className="text-sm text-slate-400 flex items-center gap-2">
                <Dots /> {msg.content}
              </span>
            ) : (
              <>
                <div className="text-slate-800"><Markdown text={msg.content} /></div>
                <SourceCards sources={msg.sources} />
                {(msg.cached || msg.elapsed) && (
                  <div className="flex gap-3 mt-2">
                    {msg.cached && (
                      <span className="text-xs text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full">⚡ cached</span>
                    )}
                    {msg.elapsed && (
                      <span className="text-xs text-slate-400">⏱ {msg.elapsed}s</span>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function WelcomeScreen({ status, cfg, accent, onQuestion, onSetup, setupState }) {
  const branding = cfg?.branding || {}
  const suggested = cfg?.suggested_questions || []
  const company = branding.company_name || 'Weld AI'
  const tagline = branding.tagline || 'Ask questions across your documents'

  // Step 1: No API key
  if (!status?.has_key) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-sm w-full text-center">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center text-2xl mx-auto mb-4 shadow-md"
               style={{ background: `linear-gradient(135deg, ${accent}, ${accent}99)` }}>🔑</div>
          <h2 className="text-xl font-bold text-slate-900 mb-2">Welcome to {company}</h2>
          <p className="text-sm text-slate-500 mb-6">{tagline}</p>
          <div className="bg-white border border-slate-200 rounded-xl p-5 text-left shadow-sm">
            <div className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold mb-3"
                 style={{ background: accent }}>1</div>
            <h3 className="font-semibold text-slate-800 mb-1">Connect your AI</h3>
            <p className="text-sm text-slate-500">Enter your Anthropic API key in the sidebar to get started.
              Get yours at <a href="https://console.anthropic.com" target="_blank" rel="noreferrer"
              className="text-blue-600 hover:underline">console.anthropic.com</a></p>
          </div>
        </div>
      </div>
    )
  }

  // Step 2: No files
  if (!status?.files?.length) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-sm w-full text-center">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center text-2xl mx-auto mb-4 shadow-md"
               style={{ background: `linear-gradient(135deg, ${accent}, ${accent}99)` }}>📂</div>
          <h2 className="text-xl font-bold text-slate-900 mb-2">Add your documents</h2>
          <p className="text-sm text-slate-500 mb-6">Copy your business documents into the docs folder on the server.</p>
          <div className="bg-white border border-slate-200 rounded-xl p-5 text-left shadow-sm">
            <div className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold mb-3"
                 style={{ background: accent }}>2</div>
            <h3 className="font-semibold text-slate-800 mb-1">Supported formats</h3>
            <p className="text-sm text-slate-500">PDF · Word · Excel · PowerPoint · CSV · Text</p>
          </div>
        </div>
      </div>
    )
  }

  // Step 3: Not indexed
  if (!status?.chunks_in_db) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-md w-full text-center">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center text-2xl mx-auto mb-4 shadow-md"
               style={{ background: `linear-gradient(135deg, ${accent}, ${accent}99)` }}>⚡</div>
          <h2 className="text-xl font-bold text-slate-900 mb-2">{status.files.length} document{status.files.length > 1 ? 's' : ''} ready to index</h2>
          <p className="text-sm text-slate-500 mb-6">Build the index once and you can ask questions in seconds.</p>
          {setupState ? (
            <SetupProgress setupState={setupState} accent={accent} />
          ) : (
            <div className="flex flex-col gap-2 w-full max-w-xs mx-auto">
              <button
                onClick={() => onSetup('full')}
                className="w-full py-3 rounded-xl text-white font-semibold text-sm transition-all hover:opacity-90 shadow-md"
                style={{ background: accent }}
              >
                ⊕ Build Index
              </button>
              <p className="text-xs text-slate-400 text-center">Takes about 1 minute · Only needed once</p>
            </div>
          )}
        </div>
      </div>
    )
  }

  // Ready — show chat welcome
  const examples = suggested.slice(0, 4)

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 gap-4">
      <div className="text-center max-w-lg">
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1">
          {status.chunks_in_db.toLocaleString()} chunks · {status.files.length} documents ·{' '}
          <span className="text-emerald-600">🟢 Ready</span>
        </p>
        <h2 className="text-xl font-bold text-slate-900 mb-1">Ask anything about your documents</h2>
        <p className="text-sm text-slate-400">Instant answers with exact source and page number cited.</p>
      </div>
      {examples.length > 0 && (
        <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
          {examples.map((q, i) => (
            <button
              key={i}
              onClick={() => onQuestion(q)}
              className="text-left bg-white border border-slate-200 hover:border-slate-300 rounded-xl p-3 text-sm text-slate-600 hover:text-slate-900 transition-all hover:shadow-sm"
            >
              💬 {q}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function SetupProgress({ setupState, accent }) {
  const pct = Math.round(setupState.progress * 100)
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 text-left shadow-sm w-full max-w-md mx-auto">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Dots />
          <span className="text-sm text-slate-600">{setupState.message}</span>
        </div>
        <span className="text-xs font-semibold tabular-nums" style={{ color: accent }}>{pct}%</span>
      </div>
      <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
        <div
          className="h-2 rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${accent}cc, ${accent})` }}
        />
      </div>
      <p className="text-xs text-slate-400 mt-2">
        {pct < 20 ? "Loading AI models..." :
         pct < 60 ? "Reading your documents..." :
         pct < 82 ? "Building semantic index..." :
         pct < 90 ? "Connecting to Claude AI..." :
         "Almost ready..."}
      </p>
    </div>
  )
}

// ─────────────────────────────────────────────
// Sidebar
// ─────────────────────────────────────────────

function Sidebar({ status, cfg, accent, onSetup, setupState, onClearCache }) {
  const branding = cfg?.branding || {}
  const company  = branding.company_name || 'Weld AI'
  const assistant = branding.assistant_name || 'Document Assistant'
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [keyError, setKeyError] = useState('')
  const [keySuccess, setKeySuccess] = useState(false)
  const [showKeyInput, setShowKeyInput] = useState(false)

  const saveKey = async () => {
    if (!apiKeyInput.startsWith('sk-ant-')) {
      setKeyError('Key must start with sk-ant-')
      return
    }
    await api.post('/api/apikey', { key: apiKeyInput })
    setKeySuccess(true)
    setKeyError('')
    setApiKeyInput('')
    setShowKeyInput(false)
    window.location.reload()
  }

  const removeKey = async () => {
    await api.del('/api/apikey')
    window.location.reload()
  }

  const kbStatus = status?.initialized
    ? 'text-emerald-600'
    : status?.chunks_in_db > 0 ? 'text-amber-600' : 'text-slate-400'

  const kbLabel = status?.initialized
    ? '🟢 Ready'
    : status?.chunks_in_db > 0 ? '🟡 Not connected' : '🔴 Not indexed'

  return (
    <aside className="w-64 flex-shrink-0 flex flex-col border-r border-slate-200 bg-slate-50 h-screen overflow-y-auto">

      {/* Brand header */}
      <div className="p-4 text-white flex-shrink-0"
           style={{ background: `linear-gradient(135deg, ${accent}dd, ${accent}99)` }}>
        <div className="font-bold text-base tracking-tight">{company}</div>
        <div className="text-xs opacity-75 flex items-center gap-1 mt-0.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_4px_#4ade80]" />
          {assistant}
        </div>
      </div>

      <div className="flex-1 flex flex-col gap-0 p-3">

        {/* Configuration */}
        <SidebarSection title="Configuration">
          <div className="text-xs text-slate-500 mb-2">
            Model: <span className="font-medium text-slate-700">{status?.model || '—'}</span>
          </div>
          {status?.key_source === 'env' && (
            <div className="text-xs text-emerald-700 bg-emerald-50 rounded-lg px-2.5 py-1.5">
              ✅ API key configured
            </div>
          )}
          {status?.key_source === 'saved' && (
            <div className="flex flex-col gap-1.5">
              <div className="text-xs text-emerald-700 bg-emerald-50 rounded-lg px-2.5 py-1.5">
                ✅ API key saved
              </div>
              <button onClick={removeKey}
                className="text-xs text-left px-1 text-slate-400 hover:text-red-500 transition-colors">
                Remove saved key
              </button>
            </div>
          )}
          {status?.key_source === 'none' && (
            <div>
              <div className="text-xs text-amber-700 bg-amber-50 rounded-lg px-2.5 py-1.5 mb-2">
                ⚠️ No API key set
              </div>
              {!showKeyInput ? (
                <button
                  onClick={() => setShowKeyInput(true)}
                  className="w-full text-xs py-1.5 rounded-lg border border-slate-200 hover:border-slate-300 text-slate-600 transition-colors"
                >
                  Enter API key
                </button>
              ) : (
                <div className="flex flex-col gap-1.5">
                  <input
                    type="password"
                    value={apiKeyInput}
                    onChange={e => { setApiKeyInput(e.target.value); setKeyError('') }}
                    placeholder="sk-ant-..."
                    className="w-full text-xs border border-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 bg-white"
                    style={{ '--tw-ring-color': accent }}
                    onKeyDown={e => e.key === 'Enter' && saveKey()}
                  />
                  {keyError && <p className="text-xs text-red-600">{keyError}</p>}
                  <button
                    onClick={saveKey}
                    className="w-full text-xs py-1.5 rounded-lg text-white font-medium transition-opacity hover:opacity-90"
                    style={{ background: accent }}
                  >
                    Save key
                  </button>
                  <a href="https://console.anthropic.com" target="_blank" rel="noreferrer"
                     className="text-xs text-center text-slate-400 hover:text-slate-600">
                    Get your key →
                  </a>
                </div>
              )}
            </div>
          )}
        </SidebarSection>

        {/* Knowledge Base */}
        <SidebarSection title={<span>Knowledge Base <span className={`text-xs normal-case font-normal ${kbStatus}`}>{kbLabel}</span></span>}>
          {status?.files?.length > 0 ? (
            <details className="group">
              <summary className="text-xs cursor-pointer text-slate-600 hover:text-slate-900 list-none flex items-center justify-between py-1">
                <span>📂 {status.files.length} document{status.files.length > 1 ? 's' : ''}
                  {status.chunks_in_db ? ` · ${status.chunks_in_db.toLocaleString()} chunks` : ''}</span>
                <span className="text-slate-400 group-open:rotate-90 transition-transform">›</span>
              </summary>
              <div className="mt-1.5 flex flex-col gap-1">
                {status.files.map((f, i) => (
                  <div key={i} className="flex items-center gap-1.5 py-1 px-2 rounded-lg bg-white border border-slate-100">
                    <span className="text-sm">{f.icon}</span>
                    <span className="text-xs text-slate-700 flex-1 truncate">{f.name}</span>
                    <FileIcon ext={f.ext} />
                  </div>
                ))}
              </div>
            </details>
          ) : (
            <p className="text-xs text-slate-400">No documents found</p>
          )}

          {/* Change detection warning */}
          {status?.changes?.has_changes && (
            <div className="mt-2 text-xs text-amber-700 bg-amber-50 rounded-lg px-2.5 py-1.5">
              ⚠️ {status.changes.new?.length || 0} new/changed file{status.changes.new?.length > 1 ? 's' : ''}
              {' '}— click Refresh
            </div>
          )}
        </SidebarSection>

        {/* Actions */}
        <SidebarSection title="Actions">
          {setupState ? (
            <div className="flex items-center gap-2 py-1.5">
              <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: accent }} />
              <span className="text-xs text-slate-500 truncate">{setupState.message}</span>
              <span className="text-xs text-slate-400 ml-auto">{Math.round(setupState.progress * 100)}%</span>
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {!status?.initialized && (
                <button
                  onClick={() => onSetup('init')}
                  className="w-full py-2 rounded-lg text-white text-xs font-semibold transition-opacity hover:opacity-90"
                  style={{ background: accent }}
                >
                  ▶ Open Assistant
                </button>
              )}
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  onClick={() => onSetup('incremental')}
                  className="py-1.5 rounded-lg text-xs font-medium border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 transition-colors"
                  title="Re-index only new or changed documents"
                >
                  ↻ Refresh
                </button>
                <button
                  onClick={() => onSetup('full')}
                  className="py-1.5 rounded-lg text-xs font-medium border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 transition-colors"
                  title="Rebuild the entire document index from scratch"
                >
                  ⊕ Build Index
                </button>
              </div>
            </div>
          )}
        </SidebarSection>

        {/* Activity */}
        {status?.stats && (
          <SidebarSection title="Activity" collapsible>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { val: status.stats.chunks_indexed, lbl: 'Chunks' },
                { val: status.stats.documents_found, lbl: 'Docs' },
                { val: status.stats.session_queries, lbl: 'Questions' },
                { val: status.stats.cache_size, lbl: 'Cached' },
              ].map(({ val, lbl }) => (
                <div key={lbl} className="bg-white border border-slate-100 rounded-lg p-2 text-center">
                  <div className="font-bold text-base leading-none" style={{ color: accent }}>{val}</div>
                  <div className="text-xs text-slate-400 mt-0.5">{lbl}</div>
                </div>
              ))}
            </div>
            {status.stats.session_cost_usd > 0 && (
              <div className="mt-1.5 text-xs text-center text-slate-400">
                💰 ${status.stats.session_cost_usd.toFixed(4)} this session
              </div>
            )}
            <p className="text-xs text-slate-400 mt-1">Last indexed: {status.stats.last_indexed}</p>
            <button
              onClick={onClearCache}
              className="mt-1.5 text-xs text-slate-400 hover:text-slate-600 transition-colors"
            >
              Clear answer cache
            </button>
          </SidebarSection>
        )}

        {/* Spacer + trust signals */}
        <div className="flex-1" />
        <div className="mt-4 pt-3 border-t border-slate-200">
          <div className="text-xs text-slate-400 space-y-1">
            <div>✓ Documents stay on your server</div>
            <div>✓ Powered by Claude AI</div>
            <div>✓ Not used for AI training</div>
          </div>
          <div className="text-xs text-slate-300 mt-2">
            Powered by <strong className="text-slate-400">WeldAI</strong> · weldai.uk
          </div>
        </div>
      </div>
    </aside>
  )
}

function SidebarSection({ title, children, collapsible = false }) {
  const [open, setOpen] = useState(true)
  return (
    <div className="mb-3">
      <button
        className="flex items-center justify-between w-full text-left mb-1.5"
        onClick={() => collapsible && setOpen(o => !o)}
      >
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</span>
        {collapsible && <span className="text-slate-300 text-xs">{open ? '▾' : '▸'}</span>}
      </button>
      {(!collapsible || open) && children}
    </div>
  )
}

// ─────────────────────────────────────────────
// Chat panel
// ─────────────────────────────────────────────

function ChatPanel({ status, cfg, accent, onSetup, setupState }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [activeTab, setActiveTab] = useState('chat')
  const [historyLoaded, setHistoryLoaded] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const cancelRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Load chat history from backend on mount
  useEffect(() => {
    if (!status?.initialized || historyLoaded) return
    api.get(`/api/history/${SESSION_ID}?limit=100`).then(data => {
      if (data.messages?.length > 0) {
        const loaded = data.messages.map((m, i) => ({
          id: i,
          role: m.role,
          content: m.content,
          sources: m.sources || [],
          elapsed: m.elapsed,
          cached: m.cached,
          streaming: false,
          status: false,
        }))
        setMessages(loaded)
      }
      setHistoryLoaded(true)
    }).catch(() => setHistoryLoaded(true))
  }, [status?.initialized, historyLoaded])

  const handleQuestion = useCallback((question) => {
    if (!question.trim() || streaming) return
    setInput('')
    const userMsg = { role: 'user', content: question, id: Date.now() }
    const thinkingMsg = { role: 'assistant', content: 'Searching documents...', status: true, id: Date.now() + 1 }
    setMessages(m => [...m, userMsg, thinkingMsg])
    setStreaming(true)

    let answerBuf = ''
    let msgId = Date.now() + 1

    // Build history from current messages for context (last 12 = 6 exchanges)
    const historyForApi = messages.slice(-12).map(m => ({
      role: m.role,
      content: m.content,
    }))

    const cancel = streamSSE('/api/ask', {
      question,
      session_id: SESSION_ID,
      history: historyForApi,
    }, (ev) => {
      if (ev.type === 'status') {
        setMessages(m => m.map(msg => msg.id === msgId
          ? { ...msg, content: ev.data, status: true }
          : msg
        ))
      } else if (ev.type === 'token') {
        answerBuf += ev.data
        setMessages(m => m.map(msg => msg.id === msgId
          ? { ...msg, content: answerBuf, status: false, streaming: true }
          : msg
        ))
      } else if (ev.type === 'cached') {
        answerBuf = ev.data
        setMessages(m => m.map(msg => msg.id === msgId
          ? { ...msg, content: answerBuf, status: false, streaming: false }
          : msg
        ))
      } else if (ev.type === 'sources') {
        setMessages(m => m.map(msg => msg.id === msgId
          ? { ...msg, sources: ev.data }
          : msg
        ))
      } else if (ev.type === 'meta') {
        setMessages(m => m.map(msg => msg.id === msgId
          ? { ...msg, streaming: false, elapsed: ev.data.elapsed, cached: ev.data.cached }
          : msg
        ))
      } else if (ev.type === 'error') {
        setMessages(m => m.map(msg => msg.id === msgId
          ? { ...msg, content: `⚠️ ${ev.data}`, status: false, streaming: false, error: true }
          : msg
        ))
        setStreaming(false)
      } else if (ev.type === 'done') {
        setMessages(m => m.map(msg => msg.id === msgId
          ? { ...msg, streaming: false }
          : msg
        ))
        setStreaming(false)
      }
    })
    cancelRef.current = cancel
  }, [streaming])

  // Allow example cards to trigger a question via custom event
  useEffect(() => {
    const handler = (e) => handleQuestion(e.detail)
    window.addEventListener('ask-question', handler)
    return () => window.removeEventListener('ask-question', handler)
  }, [handleQuestion])

  const sendInput = () => handleQuestion(input)

  return (
    <div className="flex-1 flex flex-col h-screen min-w-0">
      {/* Tab bar */}
      <div className="flex border-b border-slate-200 bg-white px-4 flex-shrink-0">
        {[
          { id: 'chat', label: '💬 Chat' },
          { id: 'search', label: '🔎 Search' },
          { id: 'about', label: 'ℹ️ About' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-current'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
            style={activeTab === tab.id ? { color: accent, borderColor: accent } : {}}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'chat' && (
        <>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
            {messages.length > 0 && historyLoaded && messages[0]?.id !== undefined && (
              <div className="flex justify-center py-2">
                <span className="text-xs text-slate-300 bg-slate-50 px-3 py-1 rounded-full border border-slate-100">
                  ↑ Conversation history
                </span>
              </div>
            )}
            {messages.length === 0 ? (
              <WelcomeScreen
                status={status}
                cfg={cfg}
                accent={accent}
                onQuestion={(q) => handleQuestion(q)}
                onSetup={onSetup}
                setupState={setupState}
              />
            ) : (
              messages.map(msg => (
                <MessageBubble key={msg.id} msg={msg} accent={accent} />
              ))
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input bar */}
          {status?.initialized && (
            <div className="flex-shrink-0 p-3 bg-white border-t border-slate-200">
              <div className="flex gap-2 bg-white border-2 border-slate-200 rounded-2xl p-2 pl-4 focus-within:border-blue-300 transition-all shadow-sm"
                   style={{ '--tw-border-opacity': 1 }}>
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendInput()}
                  placeholder="Ask a question about your documents..."
                  className="flex-1 bg-transparent text-sm text-slate-800 placeholder-slate-400 focus:outline-none"
                  disabled={streaming}
                />
                <button
                  onClick={sendInput}
                  disabled={!input.trim() || streaming}
                  className="w-9 h-9 rounded-xl flex items-center justify-center text-white transition-all disabled:opacity-30 hover:opacity-90 shadow-sm flex-shrink-0"
                  style={{ background: input.trim() && !streaming ? accent : '#94A3B8' }}
                >
                  {streaming ? (
                    <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z"/>
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <line x1="22" y1="2" x2="11" y2="13"/>
                      <polygon points="22,2 15,22 11,13 2,9"/>
                    </svg>
                  )}
                </button>
              </div>
              {messages.length > 0 && (
                <div className="flex gap-2 mt-1.5 px-1">
                  <button
                    onClick={() => {
                      setMessages([])
                      setHistoryLoaded(false)
                      api.del(`/api/history/${SESSION_ID}`)
                    }}
                    className="text-xs text-slate-400 hover:text-red-500 transition-colors"
                  >
                    🗑️ New conversation
                  </button>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {activeTab === 'search' && (
        <SearchPanel vectorSearch accent={accent} />
      )}

      {activeTab === 'about' && (
        <AboutPanel status={status} cfg={cfg} />
      )}
    </div>
  )
}

// ─────────────────────────────────────────────
// Search panel
// ─────────────────────────────────────────────

function SearchPanel({ accent }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)  // null=not searched, []=no results
  const [loading, setLoading] = useState(false)

  const search = async () => {
    if (!query.trim()) return
    setLoading(true)
    try {
      const res = await api.post('/api/search', { query, k: 6 })
      setResults(res.results || [])
    } catch (e) {
      setResults([])
    }
    setLoading(false)
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h3 className="font-semibold text-slate-800 mb-1">Direct document search</h3>
      <p className="text-sm text-slate-500 mb-4">Search your document index directly — no AI, just semantic similarity.</p>
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && search()}
          placeholder="e.g. refund policy, safety procedures..."
          className="flex-1 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 bg-white"
        />
        <button
          onClick={search}
          disabled={loading}
          className="px-5 py-2 rounded-lg text-white text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-50 flex-shrink-0"
          style={{ background: accent }}
        >
          {loading ? (
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />Searching</span>
          ) : 'Search'}
        </button>
      </div>
      {results?.map((r, i) => (
        <details key={i} className="mb-2 bg-white border border-slate-200 rounded-xl overflow-hidden">
          <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-slate-700 hover:bg-slate-50 flex items-center gap-2">
            <span>{r.icon || '📄'}</span>
            <span>{r.display_name}</span>
          </summary>
          <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-600 leading-relaxed">
            {r.content}
          </div>
        </details>
      ))}
      {results !== null && results.length === 0 && (
        <div className="text-center py-12">
          <p className="text-2xl mb-2">🔍</p>
          <p className="text-sm text-slate-500 font-medium">No matching passages found</p>
          <p className="text-xs text-slate-400 mt-1">Try different keywords or a broader search term</p>
        </div>
      )}
      {results === null && !loading && (
        <div className="text-center py-12 text-slate-300">
          <p className="text-4xl mb-3">📚</p>
          <p className="text-sm text-slate-400">Enter a search term to find passages in your documents</p>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────
// About panel
// ─────────────────────────────────────────────

function AboutPanel({ status, cfg }) {
  const stats = status?.stats
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h3 className="font-semibold text-slate-800 mb-4">
        About {cfg?.branding?.company_name || 'Weld AI'}
      </h3>
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">System</h4>
          <dl className="space-y-1.5 text-sm">
            <div className="flex justify-between">
              <dt className="text-slate-500">AI Model</dt>
              <dd className="font-medium text-slate-800">{status?.model}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-slate-500">Embeddings</dt>
              <dd className="font-medium text-slate-800">Local (fastembed)</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-slate-500">Vector DB</dt>
              <dd className="font-medium text-slate-800">PostgreSQL + pgvector</dd>
            </div>
          </dl>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Knowledge Base</h4>
          <dl className="space-y-1.5 text-sm">
            <div className="flex justify-between">
              <dt className="text-slate-500">Documents</dt>
              <dd className="font-medium text-slate-800">{stats?.documents_found ?? '—'}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-slate-500">Chunks</dt>
              <dd className="font-medium text-slate-800">{stats?.chunks_indexed?.toLocaleString() ?? '—'}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-slate-500">Last indexed</dt>
              <dd className="font-medium text-slate-800">{stats?.last_indexed ?? '—'}</dd>
            </div>
          </dl>
        </div>
      </div>
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Supported Formats</h4>
        <p className="text-sm text-slate-600">PDF · Word (.docx .doc) · PowerPoint (.pptx .ppt) · Excel (.xlsx .xls) · CSV · Plain text · RTF</p>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Root App
// ─────────────────────────────────────────────

export default function App() {
  const [status, setStatus] = useState(null)
  const [cfg, setCfg] = useState(null)
  const [setupState, setSetupState] = useState(null)
  const accent = cfg?.branding?.accent_colour || '#185FA5'

  const fetchStatus = useCallback(async () => {
    try {
      const s = await api.get('/api/status')
      setStatus(s)
    } catch (e) {
      console.error('Status fetch failed:', e)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    api.get('/api/config').then(setCfg)
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  const handleSetup = useCallback((mode) => {
    setSetupState({ progress: 0, message: 'Starting...' })
    streamSSE('/api/setup', { mode }, (ev) => {
      if (ev.progress !== undefined) {
        setSetupState({ progress: ev.progress, message: ev.message || '' })
      }
      if (ev.done) {
        setSetupState(null)
        fetchStatus()
      }
    })
  }, [fetchStatus])

  // Auto-connect if key + index exist but not initialized
  useEffect(() => {
    if (status && !status.initialized && status.has_key && status.chunks_in_db > 0 && !setupState) {
      handleSetup('init')
    }
  }, [status?.initialized, status?.has_key, status?.chunks_in_db, handleSetup])

  const handleClearCache = async () => {
    await api.del('/api/cache')
    fetchStatus()
  }

  if (!status || !cfg) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-50">
        <div className="flex flex-col items-center gap-3">
          <Dots />
          <p className="text-sm text-slate-400">Connecting...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      <Sidebar
        status={status}
        cfg={cfg}
        accent={accent}
        onSetup={handleSetup}
        setupState={setupState}
        onClearCache={handleClearCache}
      />
      <ChatPanel
        status={status}
        cfg={cfg}
        accent={accent}
        onSetup={handleSetup}
        setupState={setupState}
      />
    </div>
  )
}

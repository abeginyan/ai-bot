import { useState } from 'react'

const RAG_URL = import.meta.env.VITE_RAG_URL ?? 'http://localhost:8000'

interface Source {
  id: string
  title: string
  score: number
  snippet: string
}

interface RagResponse {
  answer: string | null
  sources: Source[]
  generated: boolean
  generationError?: string | null
}

export default function App() {
  const [query, setQuery] = useState('Кто такой Ферро-Страж?')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<RagResponse | null>(null)
  const [ragError, setRagError] = useState<string | null>(null)

  // Per-request prompt-injection defense layers (secure by default).
  const [layer1, setLayer1] = useState(true)
  const [layer2, setLayer2] = useState(true)
  const [layer3, setLayer3] = useState(true)

  const layers = [
    { on: layer1, set: setLayer1, label: 'Слой 1 — защитный системный промпт (контекст как данные)' },
    { on: layer2, set: setLayer2, label: 'Слой 2 — очистка найденных документов (скрыть инъекции)' },
    { on: layer3, set: setLayer3, label: 'Слой 3 — фильтр ответа модели' },
  ]

  async function ask(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setRagError(null)
    setResult(null)
    try {
      const res = await fetch(`${RAG_URL}/rag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, topK: 4, layer1, layer2, layer3 }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      setResult((await res.json()) as RagResponse)
    } catch (err) {
      setRagError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: 760, margin: '0 auto' }}>
      <h1>AI Bot — вселенная «Эйтрин»</h1>

      <form onSubmit={ask} style={{ display: 'flex', gap: 8, margin: '1.5rem 0' }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Задайте вопрос по вселенной…"
          style={{ flex: 1, padding: '0.6rem', fontSize: '1rem', borderRadius: 6, border: '1px solid #ccc' }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{ padding: '0.6rem 1.2rem', fontSize: '1rem', borderRadius: 6, cursor: 'pointer' }}
        >
          {loading ? '…' : 'Спросить'}
        </button>
      </form>

      <fieldset style={{ border: '1px solid #ddd', borderRadius: 8, padding: '0.75rem 1rem', margin: '0 0 1.5rem' }}>
        <legend style={{ color: '#666', fontSize: '0.9rem', padding: '0 0.4rem' }}>
          Защита от инъекций (применяется в бэкенде)
        </legend>
        {layers.map((l, i) => (
          <label key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0.25rem 0', fontSize: '0.9rem' }}>
            <input type="checkbox" checked={l.on} onChange={(e) => l.set(e.target.checked)} />
            {l.label}
          </label>
        ))}
      </fieldset>

      {ragError && <p style={{ color: 'crimson' }}>Error: {ragError}</p>}

      {result && (
        <section>
          {result.answer ? (
            <div style={{ padding: '1rem', border: '1px solid #ddd', borderRadius: 8, whiteSpace: 'pre-wrap' }}>
              {result.answer}
            </div>
          ) : (
            <p style={{ color: '#a60' }}>
              {result.generationError
                ? `Генерация недоступна: ${result.generationError} — показаны только найденные источники.`
                : 'Генерация ответа отключена (нет OPENAI_API_KEY) — показаны только найденные источники.'}
            </p>
          )}

          <h3 style={{ marginTop: '1.5rem' }}>Источники</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {result.sources.map((s) => (
              <li key={s.id} style={{ padding: '0.75rem', border: '1px solid #eee', borderRadius: 8, marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>{s.title}</strong>
                  <span style={{ color: '#888', fontSize: '0.85rem' }}>
                    {s.id} · {s.score.toFixed(3)}
                  </span>
                </div>
                <div style={{ color: '#555', fontSize: '0.9rem', marginTop: 4, whiteSpace: 'pre-wrap' }}>
                  {s.snippet}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  )
}

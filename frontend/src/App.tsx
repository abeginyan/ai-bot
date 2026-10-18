import { useEffect, useState } from 'react'

interface MessageResponse {
  message: string
  timestamp: string
}

export default function App() {
  const [data, setData] = useState<MessageResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/message')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json() as Promise<MessageResponse>
      })
      .then(setData)
      .catch((err) => setError(err.message))
  }, [])

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: 640, margin: '0 auto' }}>
      <h1>AI Bot</h1>
      <p>Frontend (React + Vite) → Backend (.NET 8)</p>

      {error && <p style={{ color: 'crimson' }}>Error: {error}</p>}

      {data ? (
        <div style={{ padding: '1rem', border: '1px solid #ccc', borderRadius: 8 }}>
          <strong>{data.message}</strong>
          <div style={{ color: '#666', fontSize: '0.9rem' }}>{data.timestamp}</div>
        </div>
      ) : (
        !error && <p>Loading…</p>
      )}
    </main>
  )
}

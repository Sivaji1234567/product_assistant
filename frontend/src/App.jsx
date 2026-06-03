import { useState } from 'react'
import SearchBox from './components/SearchBox'
import ResultCard from './components/ResultCard'
import SearchHistory from './components/SearchHistory'
import { getRecommendation } from './api'
import './App.css'

const HISTORY_KEY = 'pa_history'
const MAX_HISTORY = 5

export default function App() {
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [] }
    catch { return [] }
  })

  function addToHistory(query) {
    setHistory(prev => {
      const updated = [query, ...prev.filter(q => q !== query)].slice(0, MAX_HISTORY)
      localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
      return updated
    })
  }

  async function handleSearch(query) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await getRecommendation(query)
      setResult(data)
      addToHistory(query)
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Something went wrong. Please try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-icon">🛒</div>
        <h1>Product Assistant</h1>
        <p>Find the best platform & price for any product — live data, real prices</p>
      </header>

      <main className="main">
        <SearchBox onSearch={handleSearch} loading={loading} />
        <SearchHistory history={history} onSelect={handleSearch} />

        {error && (
          <div className="error-banner" role="alert">
            <span>⚠️</span> {error}
          </div>
        )}

        {loading && (
          <div className="loading-box">
            <div className="loading-ring" />
            <div className="loading-steps">
              <p>Scanning Amazon, Flipkart, Croma, Reliance Digital...</p>
              <p className="loading-sub">Comparing prices & ratings in real time</p>
            </div>
          </div>
        )}

        {result && !loading && <ResultCard result={result} />}

        {!result && !loading && !error && (
          <div className="suggestions">
            <p className="suggestions-title">Try searching for:</p>
            <div className="suggestion-chips">
              {['iPhone 15 128GB', 'Samsung Galaxy S24', 'Sony WH-1000XM5', 'MacBook Air M2'].map(s => (
                <button key={s} className="suggestion-chip" onClick={() => handleSearch(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

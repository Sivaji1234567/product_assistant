import { useState } from 'react'

export default function SearchBox({ onSearch, loading }) {
  const [query, setQuery] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    if (query.trim()) onSearch(query.trim())
  }

  return (
    <form className="search-box" onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="e.g. iPhone 15 128GB, Samsung Galaxy S24..."
        disabled={loading}
        className="search-input"
        aria-label="Product search"
      />
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="search-button"
      >
        {loading ? <span className="spinner" /> : 'Find Best Deal'}
      </button>
    </form>
  )
}

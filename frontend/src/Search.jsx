import React, { useState } from 'react'

function Search({ apiUrl }) {
  const [query, setQuery] = useState('')
  const [k, setK] = useState(10)
  const [metric, setMetric] = useState('cosine')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [hasSearched, setHasSearched] = useState(false)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setHasSearched(true)

    try {
      const response = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          k: k,
          metric: metric,
        }),
      })

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      const data = await response.json()
      setResults(data.results || [])
    } catch (err) {
      setError(err.message)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="search-container">
      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          className="search-input"
          placeholder="Enter your search query..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
        />
        <button type="submit" className="search-button" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      <div className="search-options">
        <label>
          Results:
          <input
            type="number"
            min="1"
            max="100"
            value={k}
            onChange={(e) => setK(parseInt(e.target.value) || 10)}
            disabled={loading}
          />
        </label>
        <label>
          Metric:
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            disabled={loading}
          >
            <option value="cosine">Cosine Similarity</option>
            <option value="dot_product">Dot Product</option>
            <option value="euclidean">Euclidean Distance</option>
          </select>
        </label>
      </div>

      {error && <div className="error">Error: {error}</div>}

      <div className="results-container">
        {loading ? (
          <div className="loading">Searching...</div>
        ) : results.length > 0 ? (
          <>
            <div className="results-header">
              Found {results.length} result{results.length !== 1 ? 's' : ''}
            </div>
            {results.map((result, index) => (
              <div key={result.id || index} className="result-item">
                <div className="result-text">{result.text}</div>
                <div className="result-meta">
                  <span>ID: {result.id}</span>
                  <span className="result-score">Score: {result.score}</span>
                </div>
              </div>
            ))}
          </>
        ) : hasSearched ? (
          <div className="loading">No results found. Try a different query.</div>
        ) : null}
      </div>
    </div>
  )
}

export default Search


import { useState } from 'react'

function Stars({ rating }) {
  const full  = Math.floor(rating)
  const half  = rating - full >= 0.5 ? 1 : 0
  const empty = 5 - full - half
  return (
    <span className="stars" aria-label={`${rating} out of 5`}>
      {'★'.repeat(full)}
      {half ? '⯨' : ''}
      {'☆'.repeat(empty)}
    </span>
  )
}

function PlatformBadge({ name }) {
  const colors = {
    Amazon:   '#FF9900',
    Flipkart: '#2874F0',
    Croma:    '#007F6E',
    Reliance: '#1C4B99',
  }
  const bg = colors[name] || '#6B7280'
  return (
    <span className="platform-badge" style={{ background: bg }}>
      {name}
    </span>
  )
}

export default function ResultCard({ result }) {
  const [showAll, setShowAll] = useState(false)
  const { recommended_platform, price, rating, reason, all_platforms } = result

  return (
    <div className="result-card">
      <div className="result-header">
        <div>
          <div className="result-label">Best Platform</div>
          <PlatformBadge name={recommended_platform} />
        </div>
        <div className="result-price-block">
          <div className="result-label">Price</div>
          <div className="result-price">₹{price.toLocaleString('en-IN')}</div>
        </div>
        <div className="result-rating-block">
          <div className="result-label">Rating</div>
          <div className="result-rating">
            <Stars rating={rating} />
            <span className="rating-value">{rating}/5</span>
          </div>
        </div>
      </div>

      <p className="result-reason">{reason}</p>

      <button className="toggle-btn" onClick={() => setShowAll(v => !v)}>
        {showAll ? 'Hide comparison ▲' : 'Compare all platforms ▼'}
      </button>

      {showAll && (
        <table className="compare-table">
          <thead>
            <tr>
              <th>Platform</th>
              <th>Price (₹)</th>
              <th>Rating</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {all_platforms.map((p) => (
              <tr key={p.name} className={p.name === recommended_platform ? 'row-winner' : ''}>
                <td><PlatformBadge name={p.name} /></td>
                <td>{p.price.toLocaleString('en-IN')}</td>
                <td>{p.rating}</td>
                <td>
                  <div className="score-bar-wrap">
                    <div className="score-bar" style={{ width: `${(p.score * 100).toFixed(0)}%` }} />
                    <span>{(p.score * 100).toFixed(1)}%</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function SearchHistory({ history, onSelect }) {
  if (!history || history.length === 0) return null
  return (
    <div className="history">
      <span className="history-label">Recent:</span>
      {history.map((item, i) => (
        <button key={i} className="history-chip" onClick={() => onSelect(item)}>
          {item}
        </button>
      ))}
    </div>
  )
}

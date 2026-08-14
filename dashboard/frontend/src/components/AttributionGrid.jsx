const cssClass = { 'GLM': 'glm', 'XGBoost': 'xgb', 'Neural Net': 'nn' };

function AttributionBar({ feature, contribution, maxAbs }) {
  const pct = maxAbs > 0 ? (Math.abs(contribution) / maxAbs) * 100 : 0;
  const isPositive = contribution > 0;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: 'var(--text)' }}>{feature}</span>
        <span style={{ color: isPositive ? '#fca5a5' : '#6ee7b7', fontVariantNumeric: 'tabular-nums' }}>
          {isPositive ? '+' : ''}{contribution.toFixed(3)}
        </span>
      </div>
      <div style={{ height: 6, background: 'var(--panel-2)', borderRadius: 3, overflow: 'hidden', position: 'relative' }}>
        <div style={{
          position: 'absolute',
          left: isPositive ? '50%' : `${50 - pct/2}%`,
          width: `${pct/2}%`,
          height: '100%',
          background: isPositive ? '#ef4444' : '#10b981',
        }} />
        <div style={{ position: 'absolute', left: '50%', width: 1, height: '100%', background: 'var(--border)' }} />
      </div>
    </div>
  );
}

export default function AttributionGrid({ predictions }) {
  return (
    <div className="section">
      <h2>Per-feature attribution (frequency model)</h2>
      <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: -8, marginBottom: 12 }}>
        Red bars increase the predicted frequency; green bars decrease it.
        GLM uses coefficient × value, XGBoost uses TreeSHAP, Neural Net uses gradient × (input − background).
      </p>
      <div className="attribution">
        {predictions.map((p) => {
          const maxAbs = Math.max(...p.attributions.map((a) => Math.abs(a.contribution)));
          return (
            <div key={p.model} className={`col ${cssClass[p.model]}`} style={{ borderTop: `3px solid var(--${cssClass[p.model]})` }}>
              <h4>{p.model}</h4>
              {p.attributions.map((a, i) => (
                <AttributionBar key={i} feature={a.feature} contribution={a.contribution} maxAbs={maxAbs} />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const eur = (v) => `€${Math.round(v).toLocaleString()}`;
const eurDecimal = (v) => `€${v.toFixed(2)}`;
const cssClass = { 'GLM': 'glm', 'XGBoost': 'xgb', 'Neural Net': 'nn' };

export default function PremiumCards({ predictions }) {
  return (
    <div className="cards">
      {predictions.map((p) => {
        const delta = p.delta_vs_portfolio;
        const sign = delta >= 0 ? 'pos' : 'neg';
        return (
          <div key={p.model} className={`card ${cssClass[p.model]}`}>
            <h3>{p.model}</h3>
            <div className="premium">{eur(p.pure_premium_eur)}</div>
            <div className={`delta ${sign}`}>
              {delta >= 0 ? '+' : ''}{eur(delta)} vs portfolio
            </div>
            <div className="breakdown">
              <div className="row"><span>Frequency rate</span><span className="val">{(p.frequency_rate * 100).toFixed(2)}%</span></div>
              <div className="row"><span>Severity (avg claim)</span><span className="val">{eur(p.severity_eur)}</span></div>
              <div className="row"><span>= Pure premium</span><span className="val">{eurDecimal(p.pure_premium_eur)}</span></div>
              <div className="row" style={{ marginTop: 6, opacity: 0.7 }}>
                <span>Portfolio mean</span><span className="val">{eur(p.portfolio_mean_pure_premium)}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

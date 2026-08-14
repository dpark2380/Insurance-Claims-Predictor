const NUMERIC_FIELDS = [
  { key: 'VehPower',   label: 'Vehicle power' },
  { key: 'VehAge',     label: 'Vehicle age (years)' },
  { key: 'DrivAge',    label: 'Driver age' },
  { key: 'BonusMalus', label: 'Bonus-Malus score' },
  { key: 'Density',    label: 'Density (people/km²)' },
  { key: 'Exposure',   label: 'Exposure (years)', step: 0.01, min: 0.01, max: 2.5 },
];
const CATEGORICAL_FIELDS = [
  { key: 'Area',     label: 'Area code' },
  { key: 'VehBrand', label: 'Vehicle brand' },
  { key: 'VehGas',   label: 'Fuel type', useDisplay: true },
  { key: 'Region',   label: 'Region' },
];

export default function PolicyForm({ schema, policy, setPolicy, onSubmit, onRandomize, loading }) {
  const update = (k) => (e) => {
    let v = e.target.value;
    if (NUMERIC_FIELDS.find((f) => f.key === k)) {
      v = k === 'Exposure' ? parseFloat(v) : parseInt(v, 10);
      if (Number.isNaN(v)) v = '';
    }
    setPolicy({ ...policy, [k]: v });
  };

  return (
    <div className="panel">
      <h2>Policy Profile</h2>

      {NUMERIC_FIELDS.map((f) => {
        const range = schema?.numeric_ranges?.[f.key];
        return (
          <div className="form-row" key={f.key}>
            <label>{f.label}</label>
            <div className="with-range">
              <input
                type="number"
                value={policy[f.key]}
                onChange={update(f.key)}
                step={f.step ?? 1}
                min={f.min ?? range?.[0]}
                max={f.max ?? range?.[1]}
              />
              {range && <span className="range-hint">{range[0]}–{range[1]}</span>}
            </div>
          </div>
        );
      })}

      {CATEGORICAL_FIELDS.map((f) => {
        const opts = (f.useDisplay ? schema?.display_vocab?.[f.key] : schema?.vocab?.[f.key]) ?? [];
        return (
          <div className="form-row" key={f.key}>
            <label>{f.label}</label>
            <select value={policy[f.key]} onChange={update(f.key)}>
              {opts.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
        );
      })}

      <button className="btn" onClick={onSubmit} disabled={loading || !schema}>
        {loading ? 'Predicting...' : 'Predict premium'}
      </button>
      <button className="btn secondary" onClick={onRandomize} disabled={loading || !schema} style={{ marginTop: 8 }}>
        Randomize profile
      </button>
    </div>
  );
}

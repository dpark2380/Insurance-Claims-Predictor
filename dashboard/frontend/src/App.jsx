import { useState, useEffect } from 'react';
import PolicyForm from './components/PolicyForm';
import PremiumCards from './components/PremiumCards';
import BandContext from './components/BandContext';
import AttributionGrid from './components/AttributionGrid';

const DEFAULT_POLICY = {
  VehPower: 6,
  VehAge: 5,
  DrivAge: 45,
  BonusMalus: 50,
  Density: 1000,
  Exposure: 1.0,
  Area: 'C',
  VehBrand: 'B1',
  VehGas: 'Regular',
  Region: 'R24',
};

export default function App() {
  const [schema, setSchema] = useState(null);
  const [policy, setPolicy] = useState(DEFAULT_POLICY);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/schema')
      .then((r) => r.json())
      .then(setSchema)
      .catch((e) => setError(`Failed to load schema: ${e.message}`));
  }, []);

  const predict = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(policy),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(`${r.status}: ${t}`);
      }
      setResult(await r.json());
    } catch (e) {
      setError(`Prediction failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const randomize = () => {
    if (!schema) return;
    const r = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
    setPolicy({
      VehPower: r(...schema.numeric_ranges.VehPower),
      VehAge: r(0, 30),
      DrivAge: r(18, 85),
      BonusMalus: pick([50, 50, 50, 60, 75, 95, 120, 175]),
      Density: Math.floor(Math.exp(r(0, 9))),
      Exposure: 1.0,
      Area: pick(schema.vocab.Area),
      VehBrand: pick(schema.vocab.VehBrand),
      VehGas: pick(schema.display_vocab.VehGas),
      Region: pick(schema.vocab.Region),
    });
    setResult(null);
  };

  return (
    <div className="app">
      <div className="header">
        <h1>Insurance Pure Premium Dashboard</h1>
        <p>
          Compare GLM, XGBoost, and Neural Net predictions for a single policy profile.
          Each model returns frequency × severity, with per-feature attribution.
        </p>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="layout">
        <PolicyForm
          schema={schema}
          policy={policy}
          setPolicy={setPolicy}
          onSubmit={predict}
          onRandomize={randomize}
          loading={loading}
        />

        <div>
          {result ? (
            <>
              <PremiumCards predictions={result.predictions} />
              <BandContext
                policy={result.profile}
                schema={schema}
                predictions={result.predictions}
              />
              <AttributionGrid predictions={result.predictions} />
            </>
          ) : (
            <div className="empty">
              {schema
                ? 'Enter a profile on the left and click Predict.'
                : 'Loading schema...'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

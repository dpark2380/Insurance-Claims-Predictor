import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const COLORS = { 'GLM': '#4C72B0', 'XGBoost': '#DD8452', 'Neural Net': '#55A868' };

function bandFor(value, edges, labels) {
  for (let i = 0; i < edges.length - 1; i++) {
    if (value > edges[i] && value <= edges[i + 1]) return labels[i];
  }
  return null;
}

function makeChartData(bands, bandStats) {
  return bands.map((b) => ({
    band: b,
    GLM: bandStats['GLM']?.[b] ?? null,
    XGBoost: bandStats['XGBoost']?.[b] ?? null,
    'Neural Net': bandStats['Neural Net']?.[b] ?? null,
  }));
}

export default function BandContext({ policy, schema, predictions }) {
  if (!schema) return null;

  const bmBands  = ['50-60','61-80','81-100','101-150','150+'];
  const ageBands = ['18-25','26-35','36-45','46-55','56-65','65+'];
  const bmEdges   = [49,60,80,100,150,230];
  const ageEdges  = [17,25,35,45,55,65,100];

  const bmCurrent  = bandFor(policy.BonusMalus, bmEdges, bmBands);
  const ageCurrent = bandFor(policy.DrivAge,    ageEdges, ageBands);

  const bmData  = makeChartData(bmBands,  schema.bands.bonus_malus);
  const ageData = makeChartData(ageBands, schema.bands.age);

  return (
    <div className="section">
      <h2>Where this profile sits in the portfolio</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <BandChart title={`BonusMalus = ${policy.BonusMalus} (band: ${bmCurrent})`}
                   data={bmData} highlight={bmCurrent} />
        <BandChart title={`DrivAge = ${policy.DrivAge} (band: ${ageCurrent})`}
                   data={ageData} highlight={ageCurrent} />
      </div>
    </div>
  );
}

function BandChart({ title, data, highlight }) {
  return (
    <div className="panel" style={{ padding: 16 }}>
      <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>{title}</div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid stroke="#2d3a47" strokeDasharray="3 3" />
          <XAxis dataKey="band" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#1a2129', border: '1px solid #2d3a47', fontSize: 12 }}
            labelStyle={{ color: '#e2e8f0' }}
            formatter={(v) => `€${Math.round(v).toLocaleString()}`}
          />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} />
          <Bar dataKey="GLM"        fill={COLORS['GLM']} />
          <Bar dataKey="XGBoost"    fill={COLORS['XGBoost']} />
          <Bar dataKey="Neural Net" fill={COLORS['Neural Net']} />
          {highlight && <ReferenceLine x={highlight} stroke="#38bdf8" strokeWidth={2} />}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

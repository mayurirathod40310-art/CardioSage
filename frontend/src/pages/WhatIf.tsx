import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, pct, useAction } from "../api";
import { C, ClassPill, DecisionStrip, ErrorNote, ModelPicker, PageHead, Panel, PatientForm, axisProps, tooltipStyle, validatePatient } from "../components";
import { useApp } from "../context";

export default function WhatIf() {
  const { patient, scenarioB, setScenarioB, schema, algorithm } = useApp();
  const [b, setB] = useState(scenarioB ?? patient);
  const [localErr, setLocalErr] = useState<Record<string, string>>({});
  const act = useAction((body: any) => api("/scenarios/compare", { body }));
  const r = act.data;
  useEffect(() => { setScenarioB(b); /* persist */ }, [b]); // eslint-disable-line

  const run = () => {
    if (!schema) return;
    const e = { ...validatePatient(b, schema) };
    setLocalErr(e);
    if (Object.keys(e).length) return;
    act.run({ baseline: patient, scenario: b, algorithm });
  };
  const changed = schema?.features.filter((f) => patient[f.name] !== b[f.name]) ?? [];
  const cmp = r ? r.feature_comparison.slice(0, 8).map((x: any) => ({ name: x.label, A: x.contribution_a, B: x.contribution_b })) : [];

  return (
    <>
      <PageHead title="What-if scenario laboratory">Scenario A is the record from the prediction studio. Edit scenario B and the backend scores both with the same saved model.</PageHead>
      <div className="note warn" style={{ marginBottom: 18 }}><strong>Changing an input is not changing a patient.</strong> Differences show how the model responds to edited numbers. Real-world factors are correlated, and the model learned associations, not causal effects.</div>
      <div className="split">
        <Panel title="Scenario B" sub={changed.length ? `${changed.length} input${changed.length > 1 ? "s" : ""} differ from A (outlined in teal)` : "Identical to A. Change something."}
          aside={<button className="btn sm" onClick={() => setB(patient)}>Reset B to A</button>}>
          <PatientForm value={b} onChange={setB} baseline={patient} errors={localErr} />
          <div className="row"><ModelPicker /><button className="btn primary" style={{ marginTop: 20 }} disabled={act.loading || changed.length === 0} onClick={run}>{act.loading ? <><span className="spin" /> Comparing…</> : "Compare scenarios"}</button></div>
          {act.error && <div style={{ marginTop: 12 }}><ErrorNote error={act.error} /></div>}
        </Panel>

        <div className="stack">
          {!r && <div className="empty"><h3>Nothing to compare yet</h3>Edit one or more inputs in scenario B, then compare. To edit scenario A, use the <a href="/predict">prediction studio</a>.</div>}
          {r && (
            <>
              <Panel>
                <div className="grid g2">
                  <div><div className="faint small">Scenario A</div><div className="verdict num">{pct(r.a.probability_class1)}</div><ClassPill c={r.a.predicted_class} /></div>
                  <div><div className="faint small">Scenario B</div><div className="verdict num" style={{ color: "var(--teal)" }}>{pct(r.b.probability_class1)}</div><ClassPill c={r.b.predicted_class} /></div>
                </div>
                <DecisionStrip p1={r.a.probability_class1} pB={r.b.probability_class1} />
                <div className="row"><span className={`pill ${r.class_flipped ? "bad" : "neutral"}`}>{r.class_flipped ? "Predicted class changes" : "Predicted class unchanged"}</span>
                  <span className="muted">Δ P(target = 1) = <b className="num" style={{ color: "var(--text)" }}>{r.probability_difference >= 0 ? "+" : ""}{(r.probability_difference * 100).toFixed(1)} points</b></span>
                  {r.experiment_id && <span className="faint small">saved as experiment #{r.experiment_id}</span>}</div>
                {r.warnings.map((w: any, i: number) => <div key={i} className="note warn" style={{ marginTop: 10 }}>{w.message}</div>)}
              </Panel>
              <Panel title="Changed inputs"><table className="tbl"><thead><tr><th>Feature</th><th>A</th><th>B</th><th className="r">Δ</th></tr></thead>
                <tbody>{r.changes.map((c: any) => <tr key={c.feature}><td>{c.label}</td><td>{c.from_display ?? c.from}{c.unit && !c.from_display ? ` ${c.unit}` : ""}</td><td>{c.to_display ?? c.to}{c.unit && !c.to_display ? ` ${c.unit}` : ""}</td><td className="r">{c.delta > 0 ? "+" : ""}{c.delta}</td></tr>)}</tbody></table></Panel>
              <Panel title="Why the model scores them differently" sub="Per-feature contribution to P(target = 1), KernelSHAP, largest differences first">
                <ResponsiveContainer width="100%" height={Math.max(240, cmp.length * 44)}>
                  <BarChart data={cmp} layout="vertical" margin={{ left: 10, right: 20 }}>
                    <CartesianGrid stroke={C.line} horizontal={false} /><XAxis type="number" {...axisProps} /><YAxis type="category" dataKey="name" width={200} {...axisProps} tick={{ fill: C.text, fontSize: 12.5 }} />
                    <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(3)} cursor={{ fill: "rgba(255,255,255,.04)" }} /><Legend />
                    <Bar dataKey="A" name="Scenario A" fill={C.amber} radius={2} /><Bar dataKey="B" name="Scenario B" fill={C.teal} radius={2} />
                  </BarChart>
                </ResponsiveContainer>
              </Panel>
              {Object.entries(r.sweeps).map(([name, pts]: any) => {
                const ch = r.changes.find((c: any) => c.feature === name);
                return (
                  <Panel key={name} title={`Sensitivity: ${ch.label}`} sub="Vary this one input over its observed range with everything else as in scenario A">
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={pts} margin={{ right: 20 }}><CartesianGrid stroke={C.line} vertical={false} />
                        <XAxis dataKey="value" type="number" domain={["dataMin", "dataMax"]} {...axisProps} /><YAxis domain={[0, 1]} {...axisProps} tickFormatter={(v) => `${v * 100}%`} />
                        <Tooltip {...tooltipStyle} formatter={(v: number) => pct(v)} labelFormatter={(l) => `${ch.label} = ${l}`} />
                        <Line dataKey="probability_class1" name="P(target = 1)" stroke={C.teal} dot={false} strokeWidth={2} type={ch.from_display ? "stepAfter" : "monotone"} />
                        <ReferenceDot x={ch.from} y={nearest(pts, ch.from)} r={6} fill={C.amber} stroke="#0a1220" /><ReferenceDot x={ch.to} y={nearest(pts, ch.to)} r={6} fill={C.text} stroke="#0a1220" />
                      </LineChart>
                    </ResponsiveContainer>
                    <div className="faint small">Amber dot: scenario A. White dot: scenario B (other inputs held at A for this curve).</div>
                  </Panel>);
              })}
              <div className="faint small">{r.interpretation}</div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
function nearest(pts: { value: number; probability_class1: number }[], v: number) {
  return pts.reduce((best, p) => (Math.abs(p.value - v) < Math.abs(best.value - v) ? p : best), pts[0]).probability_class1;
}

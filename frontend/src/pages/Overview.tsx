import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, f3, fmtDate, pct, useApi } from "../api";
import { C, ErrorNote, Loading, PageHead, Panel, Stat, axisProps, tooltipStyle } from "../components";

export default function Overview() {
  const { data, error, loading, reload } = useApi(() => api("/overview"), []);
  if (loading && !data) return <><PageHead title="Overview" /><Loading /></>;
  if (error && !data) return <><PageHead title="Overview" /><ErrorNote error={error} retry={reload} /></>;
  const d = data!; const s = d.dataset.shape; const h = d.health; const m = d.model;
  const rawShare = d.dataset.class_distribution.raw, uniq = d.dataset.class_distribution.unique;
  const dupPct = s.duplicate_rows / s.rows;
  const chart = [
    { name: "All 1,025 rows".replace("1,025", s.rows.toLocaleString()), "target = 0": rawShare[0].count, "target = 1": rawShare[1].count },
    { name: `${s.unique_rows} unique records`, "target = 0": uniq[0].count, "target = 1": uniq[1].count },
  ];
  return (
    <>
      <PageHead title="Heart disease intelligence, with the evidence attached">
        Every figure on this page is computed by the backend from the dataset on disk and the model files on disk.
      </PageHead>

      <div className="ledger" aria-label="Dataset reality check">
        <div><div className="big num">{s.rows.toLocaleString()}</div><div className="muted">rows in heart.csv</div></div>
        <div className="arrow" aria-hidden>→</div>
        <div><div className="big num teal">{s.unique_rows}</div><div className="muted">distinct records after removing {s.duplicate_rows.toLocaleString()} exact duplicates ({pct(dupPct, 0)})</div></div>
        <div style={{ flex: 1.3 }}>
          <div style={{ fontWeight: 600 }}>Why this matters</div>
          <p className="muted small" style={{ marginTop: 6 }}>If copies of a patient land on both sides of a train/test split, the test score is inflated. CardioSage de-duplicates first and trains on {m?.split.n_train ?? "—"} records, testing on {m?.split.n_test ?? "—"}.</p>
        </div>
      </div>

      <div className="grid g4" style={{ margin: "22px 0" }}>
        <Panel><Stat label="Selected model" value={m ? m.models.find((x: any) => x.key === m.selected)?.label.split(" (")[0] : "None"} hint={m ? `version ${m.version}` : "train a model"} /></Panel>
        <Panel><Stat label="Cross-validated ROC-AUC" value={m ? f3(m.models.find((x: any) => x.key === m.selected)?.cv_roc_auc) : "—"} hint="training split only" /></Panel>
        <Panel><Stat label="Held-out test accuracy" value={m ? pct(m.models.find((x: any) => x.key === m.selected)?.test_accuracy) : "—"} hint={m ? `${m.split.n_test} test records` : ""} /></Panel>
        <Panel><Stat label="Saved predictions" value={d.prediction_stats.total} hint={<Link to="/history">open history</Link>} /></Panel>
      </div>

      <div className="grid g2">
        <Panel title="Class balance" sub="Raw rows versus unique records">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={chart} layout="vertical" margin={{ left: 20, right: 20 }}>
              <CartesianGrid stroke={C.line} horizontal={false} />
              <XAxis type="number" {...axisProps} /><YAxis type="category" dataKey="name" width={140} {...axisProps} tick={{ fill: C.text, fontSize: 12.5 }} />
              <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,.04)" }} /><Legend />
              <Bar dataKey="target = 0" stackId="a" fill={C.amber} /><Bar dataKey="target = 1" stackId="a" fill={C.teal} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Data quality findings" aside={<Link to="/dataset">Explore the data</Link>}>
          {d.warnings.filter((w: any) => w.level !== "info").map((w: any) => <div key={w.code} className={`note ${w.level === "high" ? "high" : ""}`}>{w.message}</div>)}
        </Panel>
      </div>

      <Panel title="Model leaderboard" sub="Cross-validation is on the training split; test columns are a single look at 20% held-out records." className="sheet-gap" aside={<Link to="/benchmark">Open benchmarking lab</Link>}>
        {m ? (
          <div className="scroll"><table className="tbl">
            <thead><tr><th>Model</th><th className="r">CV ROC-AUC</th><th className="r">Test accuracy</th><th className="r">Test ROC-AUC</th><th className="r">Test F1</th></tr></thead>
            <tbody>{m.models.map((x: any) => (
              <tr key={x.key} className={x.key === m.selected ? "sel" : ""}>
                <td>{x.label} {x.key === m.selected && <span className="pill c1">selected</span>}</td>
                <td className="r">{f3(x.cv_roc_auc)}</td><td className="r">{pct(x.test_accuracy)}</td><td className="r">{f3(x.test_roc_auc)}</td><td className="r">{f3(x.test_f1)}</td>
              </tr>))}</tbody>
          </table></div>
        ) : <div className="note">No model has been trained yet. <Link to="/benchmark">Train one</Link>.</div>}
      </Panel>

      <div className="grid g2" style={{ marginTop: 18 }}>
        <Panel title="Recent experiments" aside={<Link to="/history">All history</Link>}>
          {d.recent_experiments.length === 0 ? <p className="muted">Nothing yet. Run a counterfactual search or a what-if comparison and it will appear here.</p> :
            d.recent_experiments.map((e: any) => (
              <div key={e.id} style={{ padding: "8px 0", borderBottom: "1px solid var(--line-soft)" }}>
                <div>{e.title}</div><div className="faint small">{e.kind} · {fmtDate(e.created_at)}</div>
              </div>))}
        </Panel>
        <Panel title="System status">
          <dl className="kv">
            <dt>Backend</dt><dd><span className={`pill ${h.status === "ok" ? "c1" : "bad"}`}>{h.status}</span> v{h.version}, up {Math.round(h.uptime_seconds)}s</dd>
            <dt>Dataset file</dt><dd>{h.dataset_available ? "found" : "missing"}</dd>
            <dt>Active model</dt><dd>{h.active_model_version ?? "none"}</dd>
            <dt>SQLite database</dt><dd>{h.database_ok ? "ready" : "unavailable"}</dd>
            <dt>SHAP explanations</dt><dd>{h.shap_available ? "available" : "unavailable (fallback in use)"}</dd>
            <dt>Training</dt><dd>{h.training.status}</dd>
          </dl>
        </Panel>
      </div>
    </>
  );
}

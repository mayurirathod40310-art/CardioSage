import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError, f3, fmtDate, pct, useApi } from "../api";
import { C, ErrorNote, Loading, MODEL_COLORS, PageHead, Panel, Tabs, axisProps, tooltipStyle } from "../components";

type Tab = "compare" | "curves" | "detail" | "versions";
const pm = (m: { mean: number; std: number }, p = false) => (p ? `${(m.mean * 100).toFixed(1)}% ± ${(m.std * 100).toFixed(1)}` : `${m.mean.toFixed(3)} ± ${m.std.toFixed(3)}`);

export default function Benchmark() {
  const { data: b, error, loading, reload } = useApi(() => api("/benchmark"), []);
  const versions = useApi(() => api("/models/versions"), []);
  const [tab, setTab] = useState<Tab>("compare");
  const [sel, setSel] = useState<string>("");
  const [job, setJob] = useState<any>(null);
  const [startErr, setStartErr] = useState<ApiError | null>(null);
  const timer = useRef<number | null>(null);

  const poll = () => {
    api("/train/status").then((s) => {
      setJob(s);
      if (s.status === "running") timer.current = window.setTimeout(poll, 1500);
      else if (s.status === "done") { reload(); versions.reload(); window.dispatchEvent(new Event("cardiosage:retrained")); }
    }).catch(() => { timer.current = window.setTimeout(poll, 3000); });
  };
  useEffect(() => { poll(); return () => { if (timer.current) clearTimeout(timer.current); }; }, []); // eslint-disable-line
  const start = async () => {
    setStartErr(null);
    try { setJob(await api("/train", { body: {} })); timer.current = window.setTimeout(poll, 1000); }
    catch (e) { setStartErr(e as ApiError); }
  };
  const running = job?.status === "running";
  const activate = async (v: string) => { await api(`/models/versions/${v}/activate`, { method: "POST" }); reload(); versions.reload(); window.dispatchEvent(new Event("cardiosage:retrained")); };

  const trainPanel = (
    <Panel title="Train and benchmark" sub="Re-runs tuning, cross-validation and held-out evaluation for every model (about one minute). A new version is saved; older versions are kept."
      aside={<button className="btn primary" onClick={start} disabled={running}>{running ? <><span className="spin" /> Training…</> : "Run benchmark"}</button>}>
      {running && <><div className="progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}><i style={{ width: `${job.progress * 100}%` }} /></div><p className="muted small" style={{ marginTop: 8 }}>{job.message}</p></>}
      {job?.status === "error" && <div className="note err">{job.error}</div>}
      {job?.status === "done" && !running && <div className="note good">Finished: version {job.version} is now active.</div>}
      {startErr && <ErrorNote error={startErr} />}
    </Panel>
  );

  if (loading && !b) return <><PageHead title="ML benchmarking laboratory" />{trainPanel}<Loading /></>;
  if (error || !b) return <><PageHead title="ML benchmarking laboratory" />{trainPanel}<div style={{ marginTop: 16 }}><ErrorNote error={error} retry={reload} /></div></>;

  const keys = Object.keys(b.models); const real = keys.filter((k) => k !== "baseline");
  const chosen = sel || b.selected; const m = b.models[chosen];
  const cvVsTest = keys.map((k) => ({ name: b.models[k].label.split(" (")[0], "CV ROC-AUC (train)": +b.models[k].cv.metrics.roc_auc.mean.toFixed(3), "Test ROC-AUC": +(b.models[k].test.roc_auc ?? 0).toFixed(3) }));
  const cm = m.test.confusion_matrix;

  return (
    <>
      <PageHead title="ML benchmarking laboratory">Five models, one leakage-safe protocol: de-duplicate, stratified 80/20 split, tune and cross-validate on the training split only, look at the test split once.</PageHead>
      <div style={{ marginBottom: 18 }}>{trainPanel}</div>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "compare", label: "Comparison" }, { id: "curves", label: "ROC and precision-recall" }, { id: "detail", label: "Model detail" }, { id: "versions", label: "Versions and checks" }]} />

      {tab === "compare" && (
        <div className="stack">
          <div className="note"><strong>Selected: {b.models[b.selected].label}.</strong> {b.selection_rule} With {b.split.n_train} training and {b.split.n_test} test records, differences of a few points between models are within noise.</div>
          <Panel title="Cross-validated performance (training split)" sub={`${b.models.baseline.cv.scheme}. Mean ± standard deviation across folds.`}>
            <div className="scroll"><table className="tbl"><thead><tr><th>Model</th><th className="r">Accuracy %</th><th className="r">Precision</th><th className="r">Recall</th><th className="r">F1</th><th className="r">ROC-AUC</th><th className="r">Avg precision</th><th className="r">Brier</th></tr></thead>
              <tbody>{keys.map((k) => { const c = b.models[k].cv.metrics; return (
                <tr key={k} className={k === b.selected ? "sel" : ""}><td>{b.models[k].label}{k === b.selected && <> <span className="pill c1">selected</span></>}</td>
                  <td className="r">{pm(c.accuracy, true)}</td><td className="r">{pm(c.precision)}</td><td className="r">{pm(c.recall)}</td><td className="r">{pm(c.f1)}</td><td className="r">{pm(c.roc_auc)}</td><td className="r">{pm(c.average_precision)}</td><td className="r">{pm(c.brier)}</td></tr>); })}</tbody></table></div>
            <p className="faint small" style={{ marginTop: 10 }}>Hyper-parameters were tuned on the same training folds, so these cross-validation numbers are slightly optimistic.</p>
          </Panel>
          <Panel title={`Held-out test performance (${b.split.n_test} records, looked at once)`}>
            <div className="scroll"><table className="tbl"><thead><tr><th>Model</th><th className="r">Accuracy</th><th className="r">Balanced acc.</th><th className="r">Precision</th><th className="r">Recall</th><th className="r">F1</th><th className="r">ROC-AUC</th><th className="r">MCC</th><th className="r">Brier</th></tr></thead>
              <tbody>{keys.map((k) => { const t = b.models[k].test; return (
                <tr key={k} className={k === b.selected ? "sel" : ""}><td>{b.models[k].label}</td><td className="r">{pct(t.accuracy)}</td><td className="r">{pct(t.balanced_accuracy)}</td><td className="r">{f3(t.precision)}</td><td className="r">{f3(t.recall)}</td><td className="r">{f3(t.f1)}</td><td className="r">{f3(t.roc_auc)}</td><td className="r">{f3(t.mcc)}</td><td className="r">{f3(t.brier)}</td></tr>); })}</tbody></table></div>
          </Panel>
          <div className="grid g2">
            <Panel title="Cross-validation versus test ROC-AUC"><ResponsiveContainer width="100%" height={260}><BarChart data={cvVsTest}><CartesianGrid stroke={C.line} vertical={false} /><XAxis dataKey="name" {...axisProps} interval={0} tick={{ fill: C.muted, fontSize: 11 }} /><YAxis domain={[0.4, 1]} {...axisProps} /><Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,.04)" }} /><Legend />
              <Bar dataKey="CV ROC-AUC (train)" fill={C.teal} radius={[3, 3, 0, 0]} /><Bar dataKey="Test ROC-AUC" fill={C.amber} radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></Panel>
            <Panel title="Small data, class balance"><p className="small">{b.split.n_train} training records ({b.split.train_class_counts["0"]} target = 0, {b.split.train_class_counts["1"]} target = 1). The classes are close to balanced, so accuracy is meaningful, but a {b.split.n_test}-record test set means one extra mistake moves accuracy by {(100 / b.split.n_test).toFixed(1)} points.</p><p className="small">The baseline row shows what predicting the majority class achieves: accuracy {pct(b.models.baseline.test.accuracy)} and ROC-AUC 0.5. Every real model clears it by a wide margin.</p></Panel>
          </div>
        </div>
      )}

      {tab === "curves" && (
        <div className="grid g2">
          <Panel title="ROC curves (test set)" sub="Sensitivity versus false-positive rate">
            <ResponsiveContainer width="100%" height={340}><LineChart margin={{ right: 14 }}><CartesianGrid stroke={C.line} /><XAxis type="number" dataKey="x" domain={[0, 1]} {...axisProps} label={{ value: "False-positive rate", fill: C.muted, dy: 18, fontSize: 12 }} height={44} /><YAxis type="number" dataKey="y" domain={[0, 1]} {...axisProps} /><Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(2)} /><Legend />
              <Line data={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} dataKey="y" name="chance" stroke={C.faint} strokeDasharray="4 4" dot={false} />
              {real.map((k) => <Line key={k} data={b.models[k].test.roc_curve} dataKey="y" name={`${b.models[k].label.split(" (")[0]} (AUC ${f3(b.models[k].test.roc_auc)})`} stroke={MODEL_COLORS[k]} dot={false} strokeWidth={2} />)}</LineChart></ResponsiveContainer>
          </Panel>
          <Panel title="Precision-recall curves (test set)" sub="Precision for target = 1 versus recall">
            <ResponsiveContainer width="100%" height={340}><LineChart margin={{ right: 14 }}><CartesianGrid stroke={C.line} /><XAxis type="number" dataKey="x" domain={[0, 1]} {...axisProps} label={{ value: "Recall", fill: C.muted, dy: 18, fontSize: 12 }} height={44} /><YAxis type="number" dataKey="y" domain={[0, 1]} {...axisProps} /><Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(2)} /><Legend />
              {real.map((k) => <Line key={k} data={b.models[k].test.pr_curve} dataKey="y" name={`${b.models[k].label.split(" (")[0]} (AP ${f3(b.models[k].test.average_precision)})`} stroke={MODEL_COLORS[k]} dot={false} strokeWidth={2} />)}</LineChart></ResponsiveContainer>
          </Panel>
          <div className="note" style={{ gridColumn: "1 / -1" }}>With {b.split.n_test} test records these curves are step-like and noisy; the gaps between models are smaller than the uncertainty. See the bootstrap intervals under “Model detail”.</div>
        </div>
      )}

      {tab === "detail" && (
        <div className="stack">
          <Panel title="Model" aside={<select aria-label="Model" value={chosen} onChange={(e) => setSel(e.target.value)} style={{ width: 260 }}>{keys.map((k) => <option key={k} value={k}>{b.models[k].label}</option>)}</select>}>
            <p className="muted small">{m.why}</p>
            <dl className="kv"><dt>Tuned parameters</dt><dd>{Object.keys(m.params).length ? Object.entries(m.params).map(([k, v]) => `${k} = ${v}`).join(", ") : "none"}</dd>
              <dt>Tuning</dt><dd>{m.tuning.method}{m.tuning.best_cv_roc_auc != null && ` (best inner-CV AUC ${f3(m.tuning.best_cv_roc_auc)})`}</dd><dt>Fit time</dt><dd>{m.fit_seconds}s</dd></dl>
          </Panel>
          <div className="grid g2">
            <Panel title="Confusion matrix (test)" sub="Rows are the recorded label, columns the model’s output">
              <div className="cm"><div className="hd" /><div className="hd">pred target = 0</div><div className="hd">pred target = 1</div>
                <div className="hd">actual 0</div><div className="ok"><b>{cm.tn}</b>true negative</div><div className="no"><b>{cm.fp}</b>false positive</div>
                <div className="hd">actual 1</div><div className="no"><b>{cm.fn}</b>false negative</div><div className="ok"><b>{cm.tp}</b>true positive</div></div>
            </Panel>
            <Panel title="Class-specific performance (test)"><table className="tbl"><thead><tr><th>Class</th><th className="r">Precision</th><th className="r">Recall</th><th className="r">F1</th><th className="r">Support</th></tr></thead>
              <tbody>{["0", "1"].map((c) => { const x = m.test.per_class[c]; return <tr key={c}><td>target = {c}</td><td className="r">{f3(x.precision)}</td><td className="r">{f3(x.recall)}</td><td className="r">{f3(x.f1)}</td><td className="r">{x.support}</td></tr>; })}</tbody></table></Panel>
          </div>
          {m.test_ci && <Panel title="Uncertainty on the test set" sub={`${m.test_ci.method}, ${m.test_ci.n_boot} resamples, 95% interval`}>
            <table className="tbl"><tbody><tr><td>Accuracy</td><td className="r">{pct(m.test.accuracy)}</td><td className="r">{pct(m.test_ci.accuracy[0], 0)} – {pct(m.test_ci.accuracy[1], 0)}</td></tr>
              <tr><td>ROC-AUC</td><td className="r">{f3(m.test.roc_auc)}</td><td className="r">{f3(m.test_ci.roc_auc[0])} – {f3(m.test_ci.roc_auc[1])}</td></tr>
              <tr><td>F1</td><td className="r">{f3(m.test.f1)}</td><td className="r">{f3(m.test_ci.f1[0])} – {f3(m.test_ci.f1[1])}</td></tr></tbody></table></Panel>}
        </div>
      )}

      {tab === "versions" && (
        <div className="stack">
          <Panel title="Leakage check" sub="Why records are de-duplicated before splitting">
            <div className="note high">A naive random split of the raw {b.leakage_check.rows.toLocaleString()} rows puts a copy of <strong>{pct(b.leakage_check.test_rows_also_in_train, 0)}</strong> of the test rows in the training set. A default random forest then scores accuracy {pct(b.leakage_check.naive_split_accuracy)} and ROC-AUC {f3(b.leakage_check.naive_split_roc_auc)}, versus {pct(b.models.random_forest.test.accuracy)} and {f3(b.models.random_forest.test.roc_auc)} under the leakage-safe protocol. {b.leakage_check.note}</div>
          </Panel>
          <Panel title="Saved model versions">
            {versions.data && <div className="scroll"><table className="tbl"><thead><tr><th>Version</th><th>Created</th><th>Selected model</th><th className="r">CV AUC</th><th className="r">Test AUC</th><th /></tr></thead>
              <tbody>{versions.data.versions.map((v: any) => <tr key={v.version} className={v.active ? "sel" : ""}><td>{v.version}</td><td>{fmtDate(v.created_at)}</td><td>{v.selected_label}</td><td className="r">{f3(v.cv_roc_auc)}</td><td className="r">{f3(v.test_roc_auc)}</td>
                <td className="r">{v.active ? <span className="pill c1">active</span> : <button className="btn sm" onClick={() => activate(v.version)}>Activate</button>}</td></tr>)}</tbody></table></div>}
          </Panel>
          <Panel title="Environment"><dl className="kv"><dt>Dataset SHA-256</dt><dd style={{ fontSize: ".82rem" }}>{b.dataset.sha256}</dd><dt>Libraries</dt><dd>Python {b.environment.python}, scikit-learn {b.environment.scikit_learn}, NumPy {b.environment.numpy}, pandas {b.environment.pandas}</dd><dt>Training time</dt><dd>{b.train_seconds}s</dd></dl></Panel>
        </div>
      )}
    </>
  );
}

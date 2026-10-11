import { useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError, pct, useAction } from "../api";
import { ClassPill, ContribChart, DecisionStrip, Disclaimer, ErrorNote, ModelPicker, PageHead, Panel, PatientForm, validatePatient } from "../components";
import { useApp } from "../context";

export default function Predict() {
  const { patient, setPatient, schema, algorithm } = useApp();
  const [errs, setErrs] = useState<Record<string, string>>({});
  const [sample, setSample] = useState<number | null>(null);
  const act = useAction((body: any) => api("/predict", { body }));
  const r = act.data;

  const submit = async () => {
    if (!schema) return;
    const local = validatePatient(patient, schema);
    setErrs(local);
    if (Object.keys(local).length) return;
    const res = await act.run({ features: patient, algorithm });
    if (!res) setErrs((act.error as ApiError | null)?.fields ?? {});
  };
  const loadReal = async () => {
    const seed = Math.floor(Math.random() * 100000);
    const [s] = await api<any[]>(`/dataset/samples?n=1&seed=${seed}`);
    setPatient(s.features); setSample(s.target); setErrs({}); act.reset();
  };

  return (
    <>
      <PageHead title="Prediction studio">Enter a record using the dataset’s 13 columns. The saved model returns a class, a probability, and the features that pushed it there.</PageHead>
      <div className="split">
        <Panel title="Patient record" aside={<div className="row"><button className="btn sm" onClick={loadReal}>Load a real record</button></div>}>
          {sample !== null && <div className="note" style={{ marginBottom: 14 }}>This is a real, de-duplicated record from the dataset. Its recorded label is <ClassPill c={sample} />.</div>}
          <PatientForm value={patient} onChange={(p) => { setPatient(p); setErrs({}); }} errors={errs} />
          <div className="row" style={{ marginTop: 6 }}>
            <button className="btn primary" disabled={act.loading} onClick={submit}>{act.loading ? <><span className="spin" /> Predicting…</> : "Run prediction"}</button>
          </div>
          {act.error && <div style={{ marginTop: 14 }}><ErrorNote error={act.error} /></div>}
        </Panel>

        <div className="stack">
          <Panel title="Model" aside={<ModelPicker />}><p className="muted small" style={{ margin: 0 }}>Predictions always come from the persisted pipeline for the chosen model; nothing is pre-computed.</p></Panel>
          {!r && !act.loading && <div className="empty"><h3>No prediction yet</h3>Fill in the record and choose “Run prediction”. Each run is saved to local <Link to="/history">history</Link> (inputs only, no identifiers).</div>}
          {act.loading && <div className="panel"><span className="spin" /> Computing prediction and explanation…</div>}
          {r && (
            <>
              <Panel>
                <div className="row spread" style={{ alignItems: "flex-start" }}>
                  <div className="verdict">Model output: <ClassPill c={r.predicted_class} /><small>{pct(r.probability_class1)} probability of target = 1 · {r.algorithm_label} · {r.model_version}</small></div>
                  {r.saved_id && <span className="pill neutral">saved #{r.saved_id}</span>}
                </div>
                <DecisionStrip p1={r.probability_class1} />
                <div className="note">What this is: the class a statistical model outputs for these numbers. What it is not: a clinical diagnosis. In this dataset copy, <strong>target = 0 behaves like “disease present”</strong> and target = 1 like “absent” (see <Link to="/methodology">methodology</Link>).</div>
                {r.warnings.map((w: any, i: number) => <div key={i} className="note warn">{w.message}</div>)}
              </Panel>

              <Panel title="Strongest contributing features" sub={r.explanation ? `${r.explanation.method === "kernel_shap" ? "KernelSHAP" : "Substitution fallback"} on the probability of target = 1. Base value ${r.explanation.base_value.toFixed(3)} → prediction ${r.explanation.prediction.toFixed(3)}.` : undefined}>
                {r.explanation?.warning && <div className="note warn" style={{ marginBottom: 10 }}>{r.explanation.warning}</div>}
                {r.explanation && <ContribChart items={r.explanation.contributions.slice(0, 8)} />}
                {r.explanation && <p className="muted small" style={{ marginTop: 8 }}>{r.explanation.reading_guide} Contributions sum to prediction − base value (error {r.explanation.additivity_error?.toFixed(4)}).</p>}
              </Panel>

              <Panel title="Do the models agree?" sub="Same record, every trained model">
                <table className="tbl"><thead><tr><th>Model</th><th className="r">P(target = 1)</th><th>Class</th></tr></thead>
                  <tbody>{r.model_agreement.map((m: any) => <tr key={m.algorithm}><td>{m.label}</td><td className="r">{pct(m.probability_class1)}</td><td><ClassPill c={m.predicted_class} /></td></tr>)}</tbody></table>
              </Panel>

              <Panel title="How reliable is this model?">
                <p className="small muted">Cross-validated ROC-AUC {r.model_context.cv_roc_auc.toFixed(3)} ± {r.model_context.cv_roc_auc_std.toFixed(3)}; test accuracy {pct(r.model_context.test_accuracy)} on only {r.model_context.n_test} held-out records{r.model_context.test_accuracy_ci ? ` (95% bootstrap interval ${pct(r.model_context.test_accuracy_ci[0], 0)}–${pct(r.model_context.test_accuracy_ci[1], 0)})` : ""}.</p>
                <div className="row"><Link className="btn sm" to="/counterfactual">Find counterfactuals</Link><Link className="btn sm" to="/what-if">Open in what-if lab</Link><Link className="btn sm" to="/explain">Explanation center</Link></div>
              </Panel>
              <Disclaimer>{r.disclaimer}</Disclaimer>
            </>
          )}
        </div>
      </div>
    </>
  );
}

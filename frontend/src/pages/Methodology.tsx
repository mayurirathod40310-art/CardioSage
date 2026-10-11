import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, f3, pct, useApi } from "../api";
import { C, ErrorNote, Loading, PageHead, Panel, axisProps, tooltipStyle } from "../components";

export default function Methodology() {
  const ds = useApi(() => api("/dataset/summary"), []);
  const rel = useApi(() => api("/reliability"), []);
  const d = ds.data, r = rel.data;
  return (
    <>
      <PageHead title="Methodology and limitations">How CardioSage was built, what was actually measured, and what it cannot tell you.</PageHead>
      <div className="stack">
        <Panel title="Intended use"><p>CardioSage is a research and teaching tool for studying how tabular machine-learning models behave on a classic heart-disease dataset. It is not a medical device, has not been clinically validated, and must not inform decisions about any person’s health.</p></Panel>

        <Panel title="Dataset and provenance">
          {ds.loading && !d ? <Loading /> : ds.error ? <ErrorNote error={ds.error} /> : d && (<>
            <p>{d.source.name}. Original citation: {d.source.original_citation} Licence: {d.source.original_license}.</p>
            <p className="muted small">{d.source.retrieval_note}</p>
            <p>The file holds {d.shape.rows.toLocaleString()} rows but only {d.shape.unique_rows} distinct records. It pools four hospital sites (Cleveland, Hungary, Switzerland, Long Beach), so it describes historical referral patients, not a general population.</p>
            <div className={`note ${d.label_direction.verdict === "inverted" ? "high" : ""}`}><strong>Label direction.</strong> {d.label_direction.explanation} For that reason the app always says “target = 0” or “target = 1” rather than “disease” or “healthy”.</div>
            <div className="note" style={{ marginTop: 8 }}><strong>Coding caveats.</strong> The values ca = 4 and thal = 0 are not documented results and probably encode missing data; they are kept as observed. Category codes for thal differ between redistributions, so they are left unlabelled.</div>
          </>)}
        </Panel>

        <Panel title="Pipeline">
          <ol className="small" style={{ paddingLeft: 20, lineHeight: 1.8, margin: 0 }}>
            <li>Load the CSV, map known column-name variants, convert multi-class targets to binary, flag impossible values as missing.</li>
            <li>Remove exact duplicate rows, then split 80/20, stratified by class, with a fixed seed.</li>
            <li>Inside each model’s scikit-learn pipeline: median/mode imputation, standard scaling for numeric and ordinal features, one-hot encoding for categorical ones. These steps are fitted only on training folds.</li>
            <li>Tune hyper-parameters with 5-fold stratified grid search (ROC-AUC) on the training split, then estimate performance with 5-fold × 2 repeated cross-validation on the training split.</li>
            <li>Select the model with the highest cross-validated ROC-AUC. The test split plays no part in this choice.</li>
            <li>Evaluate once on the test split: accuracy, precision, recall, F1, ROC-AUC, average precision, MCC, Brier score, confusion matrix, curves, and bootstrap intervals.</li>
            <li>Persist every fitted pipeline, the metadata and the training sample as a versioned folder; the API only ever predicts with those files.</li>
          </ol>
        </Panel>

        <Panel title="Reliability evidence">
          {rel.loading && !r ? <Loading /> : rel.error ? <ErrorNote error={rel.error} /> : r && (
            <div className="stack">
              <p className="small">Shown for <b>{r.label}</b>, model version {r.model_version}.</p>
              <div className="note">{r.selection.text}</div>
              <table className="tbl"><thead><tr><th>Model ranked by CV ROC-AUC</th><th className="r">Mean</th><th className="r">Fold SD</th></tr></thead><tbody>{r.selection.ranking.map((x: any) => <tr key={x.algorithm}><td>{x.algorithm}</td><td className="r">{f3(x.cv_roc_auc_mean)}</td><td className="r">{f3(x.cv_roc_auc_std)}</td></tr>)}</tbody></table>
              <div className="grid g2">
                <div><h3 style={{ marginBottom: 8 }}>Calibration on the test set</h3>
                  <ResponsiveContainer width="100%" height={240}><LineChart data={r.calibration} margin={{ right: 14 }}><CartesianGrid stroke={C.line} /><XAxis type="number" dataKey="predicted" domain={[0, 1]} {...axisProps} label={{ value: "Mean predicted probability", fill: C.muted, dy: 18, fontSize: 12 }} height={44} /><YAxis domain={[0, 1]} {...axisProps} /><Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(2)} />
                    <Line data={[{ predicted: 0, observed: 0 }, { predicted: 1, observed: 1 }]} dataKey="observed" stroke={C.faint} strokeDasharray="4 4" dot={false} name="perfect" /><Line dataKey="observed" stroke={C.teal} strokeWidth={2} name="observed share of target = 1" /></LineChart></ResponsiveContainer>
                  <p className="faint small">Five quantile bins of {r.test.n} records: each point averages about {Math.round(r.test.n / 5)} patients, so this is a rough sanity check, not a calibration guarantee. Brier score {f3(r.test.brier)}. No recalibration was applied.</p></div>
                <div><h3 style={{ marginBottom: 8 }}>Test uncertainty</h3>
                  {r.test_ci && <table className="tbl"><tbody><tr><td>Accuracy</td><td className="r">{pct(r.test.accuracy)}</td><td className="r">{pct(r.test_ci.accuracy[0], 0)}–{pct(r.test_ci.accuracy[1], 0)}</td></tr><tr><td>ROC-AUC</td><td className="r">{f3(r.test.roc_auc)}</td><td className="r">{f3(r.test_ci.roc_auc[0])}–{f3(r.test_ci.roc_auc[1])}</td></tr></tbody></table>}
                  <h3 style={{ margin: "16px 0 8px" }}>Class-specific (test)</h3>
                  <table className="tbl"><thead><tr><th>Class</th><th className="r">Precision</th><th className="r">Recall</th><th className="r">n</th></tr></thead><tbody>{["0", "1"].map((c) => <tr key={c}><td>target = {c}</td><td className="r">{f3(r.test.per_class[c].precision)}</td><td className="r">{f3(r.test.per_class[c].recall)}</td><td className="r">{r.test.per_class[c].support}</td></tr>)}</tbody></table></div>
              </div>
            </div>)}
        </Panel>

        {r && <Panel title="What has not been evaluated" sub="CardioSage makes no claims about these">
          <div className="grid g3">{Object.entries(r.evidence_not_available).filter(([k]) => k !== "note").map(([k]) => <div key={k} className="note">{k.replace(/_/g, " ")}: <strong>not evaluated</strong></div>)}</div>
          <p className="muted small" style={{ marginTop: 12 }}>{r.evidence_not_available.note}</p></Panel>}

        {r && <Panel title="Known limitations"><ul className="small" style={{ paddingLeft: 20, lineHeight: 1.7, margin: 0 }}>{r.limitations.map((l: string) => <li key={l}>{l}</li>)}</ul></Panel>}

        <Panel title="Counterfactuals and what-if scenarios">
          <p className="small">A counterfactual is an input the fitted model classifies differently. The search tries every single-feature change, all feature pairs, and triples over the most influential features, within the value ranges seen in training data and with sex and age locked by default. Survivors are pruned to remove unnecessary changes, shrunk toward the original values, and re-scored one by one with the saved model; any that fail are discarded.</p>
          <p className="small">These are statements about the model, not causal effects or medical recommendations. Many inputs move together in real patients (for example age and maximum heart rate), so a single-feature edit can create a combination that is rare or impossible in practice. The “unusual combination” tag marks candidates farther from any real training patient than 95% of training patients are from their nearest neighbour.</p>
        </Panel>

        <Panel title="Open API"><p className="small" style={{ margin: 0 }}>Every action in this interface calls the documented REST API. Interactive documentation lives at <a href="/docs">/docs</a> on the backend (http://127.0.0.1:8000/docs).</p></Panel>
      </div>
    </>
  );
}

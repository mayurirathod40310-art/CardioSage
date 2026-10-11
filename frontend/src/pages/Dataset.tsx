import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";
import { api, f3, pct, useApi } from "../api";
import { C, ErrorNote, Loading, PageHead, Panel, Stat, Tabs, axisProps, tooltipStyle } from "../components";

type Tab = "summary" | "features" | "correlation" | "quality";

function heatColor(v: number) {
  const a = Math.min(1, Math.abs(v));
  return v >= 0 ? `rgba(63,193,176,${0.1 + a * 0.85})` : `rgba(227,165,83,${0.1 + a * 0.85})`;
}

export default function Dataset() {
  const { data: d, error, loading, reload } = useApi(() => api("/dataset/summary"), []);
  const [tab, setTab] = useState<Tab>("summary");
  const [feat, setFeat] = useState("age");
  const [method, setMethod] = useState<"pearson" | "spearman">("pearson");
  if (loading && !d) return <><PageHead title="Dataset explorer" /><Loading /></>;
  if (error || !d) return <><PageHead title="Dataset explorer" /><ErrorNote error={error} retry={reload} /></>;

  const f = d.features[feat];
  const cols: string[] = d.correlation.columns;
  const mat: number[][] = d.correlation[method];
  return (
    <>
      <PageHead title="Dataset explorer">Statistics are computed on the de-duplicated records unless a table says “raw”. Class names follow the file’s <code>target</code> column.</PageHead>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "summary", label: "Summary" }, { id: "features", label: "Feature distributions" }, { id: "correlation", label: "Correlation" }, { id: "quality", label: "Quality and provenance" }]} />

      {tab === "summary" && (
        <div className="stack">
          <div className="grid g4">
            <Panel><Stat label="Rows in file" value={d.shape.rows.toLocaleString()} /></Panel>
            <Panel><Stat label="Unique records" value={d.shape.unique_rows} hint={`${d.shape.duplicate_rows} duplicates`} /></Panel>
            <Panel><Stat label="Columns" value={d.shape.columns} hint="13 features + target" /></Panel>
            <Panel><Stat label="Missing cells" value={d.missing_total} hint="before in-pipeline imputation" /></Panel>
          </div>
          <div className="grid g2">
            <Panel title="Target class distribution">
              <table className="tbl"><thead><tr><th>Class</th><th className="r">Raw rows</th><th className="r">Unique records</th><th className="r">Share (unique)</th></tr></thead>
                <tbody>{d.class_distribution.unique.map((u: any, i: number) => (
                  <tr key={u.class}><td>target = {u.class}</td><td className="r">{d.class_distribution.raw[i].count}</td><td className="r">{u.count}</td><td className="r">{pct(u.share)}</td></tr>))}</tbody></table>
              <div className="bar-track" aria-hidden>{d.class_distribution.unique.map((u: any) => <i key={u.class} style={{ width: `${u.share * 100}%`, background: u.class ? C.teal : C.amber }} />)}</div>
            </Panel>
            <Panel title="Duplicate detection">
              <dl className="kv">
                <dt>Exact duplicate rows</dt><dd className="num">{d.duplicate_summary.duplicate_rows}</dd>
                <dt>Rows that have a twin</dt><dd className="num">{d.duplicate_summary.rows_with_a_duplicate}</dd>
                <dt>Most copies of one record</dt><dd className="num">{d.duplicate_summary.max_copies_of_one_row}</dd>
              </dl>
              <p className="muted small" style={{ marginTop: 12 }}>Duplicates are removed before any split so no record can sit in both the training and test sets.</p>
            </Panel>
          </div>
          <Panel title="Missing values by column"><div className="scroll"><table className="tbl"><tbody>
            <tr>{Object.keys(d.missing_by_column).map((k) => <th key={k}>{k}</th>)}</tr>
            <tr>{Object.values(d.missing_by_column).map((v: any, i) => <td key={i} className="num">{v}</td>)}</tr></tbody></table></div></Panel>
          <Panel title="First rows of the file (raw)"><div className="scroll"><table className="tbl"><thead><tr>{Object.keys(d.preview[0]).map((k) => <th key={k}>{k}</th>)}</tr></thead>
            <tbody>{d.preview.map((r: any, i: number) => <tr key={i}>{Object.values(r).map((v: any, j) => <td key={j} className="num">{String(v)}</td>)}</tr>)}</tbody></table></div></Panel>
        </div>
      )}

      {tab === "features" && (
        <div className="split" style={{ gridTemplateColumns: "250px 1fr" }}>
          <Panel title="Features">
            {Object.values(d.features).map((x: any) => (
              <button key={x.name} className={`btn ghost sm`} style={{ width: "100%", justifyContent: "space-between", marginBottom: 4, borderColor: x.name === feat ? C.teal : undefined, color: x.name === feat ? C.teal : undefined }} onClick={() => setFeat(x.name)}>
                <span>{x.label}</span><span className="faint">{x.kind}</span></button>))}
          </Panel>
          <div className="stack">
            <Panel title={f.label} sub={`${f.kind}${f.unit ? ` · ${f.unit}` : ""} · ${f.n_unique} distinct values · ${f.missing} missing`}>
              {f.kind === "numeric" ? (
                <>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={f.histogram}><CartesianGrid stroke={C.line} vertical={false} />
                      <XAxis dataKey="bin" {...axisProps} /><YAxis {...axisProps} /><Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,.04)" }} /><Legend />
                      <Bar dataKey="class0" name="target = 0" fill={C.amber} /><Bar dataKey="class1" name="target = 1" fill={C.teal} /></BarChart>
                  </ResponsiveContainer>
                  <div className="grid g4" style={{ marginTop: 16 }}>
                    <Stat label="Mean ± SD" value={`${f.mean.toFixed(1)} ± ${f.std.toFixed(1)}`} />
                    <Stat label="Median (Q1–Q3)" value={f.median} hint={`${f.q1}–${f.q3}`} />
                    <Stat label="Range" value={`${f.min}–${f.max}`} />
                    <Stat label="Skewness" value={f.skew.toFixed(2)} />
                  </div>
                  <div className={`note ${f.outliers.count ? "warn" : ""}`} style={{ marginTop: 14 }}>
                    <strong>{f.outliers.count} outlier{f.outliers.count === 1 ? "" : "s"}</strong> outside the 1.5×IQR fences ({f.outliers.lower_fence.toFixed(1)} to {f.outliers.upper_fence.toFixed(1)}).
                    {f.outliers.values.length > 0 && <> Values: {f.outliers.values.join(", ")}.</>} They are kept: they are plausible measurements, not errors.
                  </div>
                </>
              ) : (
                <>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={f.categories}><CartesianGrid stroke={C.line} vertical={false} />
                      <XAxis dataKey="label" {...axisProps} interval={0} /><YAxis {...axisProps} /><Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,.04)" }} /><Legend />
                      <Bar dataKey="class0" name="target = 0" fill={C.amber} /><Bar dataKey="class1" name="target = 1" fill={C.teal} /></BarChart>
                  </ResponsiveContainer>
                  <table className="tbl" style={{ marginTop: 12 }}><thead><tr><th>Value</th><th className="r">Records</th><th className="r">Share with target = 1</th></tr></thead>
                    <tbody>{f.categories.map((c: any) => <tr key={c.value}><td>{c.value} · {c.label}</td><td className="r">{c.count}</td><td className="r">{pct(c.rate_class1, 0)}</td></tr>)}</tbody></table>
                </>
              )}
            </Panel>
          </div>
        </div>
      )}

      {tab === "correlation" && (
        <div className="stack">
          <Panel title="Correlation matrix" aside={<div className="seg">{(["pearson", "spearman"] as const).map((m) => <button key={m} className={method === m ? "on" : ""} onClick={() => setMethod(m)}>{m}</button>)}</div>}
            sub="Teal is positive, amber is negative. Correlation describes the data, not the model and not causes.">
            <div className="scroll"><div className="heat" style={{ gridTemplateColumns: `70px repeat(${cols.length}, minmax(30px, 1fr))`, minWidth: 560 }}>
              <div className="h" />{cols.map((c) => <div key={c} className="h">{c}</div>)}
              {cols.map((r, i) => [<div key={r} className="h" style={{ justifyContent: "end", paddingRight: 6, aspectRatio: "auto" }}>{r}</div>, ...cols.map((c, j) => (
                <div key={r + c} title={`${r} vs ${c}: ${mat[i][j].toFixed(3)}`} style={{ background: heatColor(mat[i][j]) }}>{mat[i][j].toFixed(1)}</div>))])}
            </div></div>
          </Panel>
          <div className="grid g2">
            <Panel title="Association with target" sub="Pearson (point-biserial for the binary target)">
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={d.target_correlation} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" domain={[-0.6, 0.6]} {...axisProps} /><YAxis type="category" dataKey="feature" width={70} {...axisProps} tick={{ fill: C.text, fontSize: 12.5 }} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(3)} cursor={{ fill: "rgba(255,255,255,.04)" }} />
                  <Bar dataKey="pearson" radius={3}>{d.target_correlation.map((r: any) => <Cell key={r.feature} fill={r.pearson >= 0 ? C.teal : C.amber} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </Panel>
            <Panel title="Strongest feature-to-feature links">
              <table className="tbl"><thead><tr><th>Pair</th><th className="r">Pearson</th></tr></thead><tbody>{d.top_feature_pairs.map((p: any) => <tr key={p.a + p.b}><td>{p.a} · {p.b}</td><td className="r">{f3(p.pearson)}</td></tr>)}</tbody></table>
              <p className="muted small" style={{ marginTop: 12 }}>Correlated inputs mean that editing one feature in a what-if scenario while holding others fixed can create combinations that rarely occur in real patients.</p>
            </Panel>
          </div>
        </div>
      )}

      {tab === "quality" && (
        <div className="stack">
          <Panel title="Warnings">{d.warnings.map((w: any) => <div key={w.code} className={`note ${w.level === "high" ? "high" : ""}`}><strong>{w.level}</strong> · {w.message}</div>)}</Panel>
          <Panel title="Is the label direction as documented?" sub="Computed from the data: correlation of each marker with target">
            <div className={`note ${d.label_direction.verdict === "inverted" ? "high" : "good"}`}>{d.label_direction.explanation}</div>
            <table className="tbl" style={{ marginTop: 12 }}><thead><tr><th>Marker</th><th className="r">Correlation with target</th></tr></thead>
              <tbody>{Object.entries(d.label_direction.correlations_with_target).map(([k, v]: any) => <tr key={k}><td>{k}</td><td className="r">{f3(v)}</td></tr>)}</tbody></table>
          </Panel>
          <Panel title="Detected feature types"><div className="scroll"><table className="tbl"><thead><tr><th>Feature</th><th>Type</th><th className="r">Distinct</th><th className="r">Outliers</th></tr></thead>
            <tbody>{Object.values(d.features).map((x: any) => <tr key={x.name}><td>{x.label}</td><td>{x.kind}</td><td className="r">{x.n_unique}</td><td className="r">{x.outliers ? x.outliers.count : "n/a"}</td></tr>)}</tbody></table></div></Panel>
          <Panel title="Provenance and licence">
            <dl className="kv">
              <dt>Dataset</dt><dd>{d.source.name}</dd><dt>Original citation</dt><dd>{d.source.original_citation}</dd><dt>Original licence</dt><dd>{d.source.original_license}</dd>
              <dt>Retrieved from</dt><dd><a href={d.source.retrieved_from}>{d.source.retrieved_from}</a></dd><dt>SHA-256</dt><dd style={{ fontSize: ".82rem" }}>{d.file.sha256}</dd>
            </dl>
            <div className="note" style={{ marginTop: 12 }}>{d.source.retrieval_note}</div>
          </Panel>
        </div>
      )}
    </>
  );
}

import { Fragment, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, fmtDate, pct, useApi } from "../api";
import { ClassPill, Empty, ErrorNote, Loading, PageHead, Panel, Tabs } from "../components";
import { useApp } from "../context";

type Tab = "predictions" | "experiments" | "privacy";
const PAGE = 15;

export default function History() {
  const [tab, setTab] = useState<Tab>("predictions");
  return (
    <>
      <PageHead title="Prediction and experiment history">Everything is stored locally in SQLite on this computer. Inputs are stored; names and identifiers never are.</PageHead>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "predictions", label: "Predictions" }, { id: "experiments", label: "Experiments" }, { id: "privacy", label: "What is stored" }]} />
      {tab === "predictions" && <Predictions />}{tab === "experiments" && <Experiments />}{tab === "privacy" && <Privacy />}
    </>
  );
}

function Predictions() {
  const nav = useNavigate(); const { setPatient } = useApp();
  const [cls, setCls] = useState(""); const [sort, setSort] = useState("created_at"); const [order, setOrder] = useState("desc");
  const [minP, setMinP] = useState(""); const [maxP, setMaxP] = useState(""); const [page, setPage] = useState(0);
  const [open, setOpen] = useState<number | null>(null); const [confirm, setConfirm] = useState(false);
  const q = new URLSearchParams({ limit: String(PAGE), offset: String(page * PAGE), sort, order });
  if (cls) q.set("predicted_class", cls); if (minP) q.set("min_probability", minP); if (maxP) q.set("max_probability", maxP);
  const { data, error, loading, reload } = useApi(() => api(`/history/predictions?${q}`), [q.toString()]);
  const del = async (id: number) => { await api(`/history/predictions/${id}`, { method: "DELETE" }); reload(); };
  const clear = async () => { await api("/history/predictions", { method: "DELETE" }); setConfirm(false); setPage(0); reload(); };
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1;
  return (
    <Panel title={`Predictions${data ? ` (${data.total})` : ""}`} aside={data && data.total > 0 && (confirm
      ? <div className="row"><span className="small">Delete all {data.total} saved predictions?</span><button className="btn danger sm" onClick={clear}>Yes, delete all</button><button className="btn sm" onClick={() => setConfirm(false)}>Cancel</button></div>
      : <button className="btn danger sm" onClick={() => setConfirm(true)}>Delete all</button>)}>
      <div className="grid g4" style={{ marginBottom: 16 }}>
        <div><label className="f" htmlFor="hc">Predicted class</label><select id="hc" value={cls} onChange={(e) => { setCls(e.target.value); setPage(0); }}><option value="">Any</option><option value="0">target = 0</option><option value="1">target = 1</option></select></div>
        <div><label className="f" htmlFor="hmin">Min P(target = 1)</label><input id="hmin" type="number" min={0} max={1} step={0.05} value={minP} onChange={(e) => { setMinP(e.target.value); setPage(0); }} /></div>
        <div><label className="f" htmlFor="hmax">Max P(target = 1)</label><input id="hmax" type="number" min={0} max={1} step={0.05} value={maxP} onChange={(e) => { setMaxP(e.target.value); setPage(0); }} /></div>
        <div><label className="f" htmlFor="hs">Sort</label><div className="row" style={{ flexWrap: "nowrap" }}><select id="hs" value={sort} onChange={(e) => setSort(e.target.value)}><option value="created_at">Date</option><option value="probability">Probability</option><option value="algorithm">Model</option></select>
          <button className="btn" aria-label="Toggle sort order" onClick={() => setOrder(order === "desc" ? "asc" : "desc")}>{order === "desc" ? "↓" : "↑"}</button></div></div>
      </div>
      {loading && !data ? <Loading /> : error ? <ErrorNote error={error} retry={reload} /> : data!.total === 0 ? <Empty title="No saved predictions">{cls || minP || maxP ? "No predictions match these filters." : "Run a prediction in the studio and it will be listed here."}</Empty> : (
        <>
          <div className="scroll"><table className="tbl"><thead><tr><th>When</th><th>Model</th><th>Output</th><th className="r">P(target = 1)</th><th>Top feature</th><th /></tr></thead>
            <tbody>{data!.items.map((p: any) => (
              <Fragment key={p.id}>
                <tr><td>{fmtDate(p.created_at)}</td><td>{p.algorithm}<div className="faint small">{p.model_version}</div></td><td><ClassPill c={p.predicted_class} /></td><td className="r">{pct(p.probability_class1)}</td>
                  <td>{p.top_features[0]?.feature ?? "—"}</td>
                  <td className="r"><div className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}><button className="btn sm" onClick={() => setOpen(open === p.id ? null : p.id)} aria-expanded={open === p.id}>{open === p.id ? "Hide" : "View"}</button><button className="btn danger sm" onClick={() => del(p.id)} aria-label={`Delete prediction ${p.id}`}>Delete</button></div></td></tr>
                {open === p.id && <tr><td colSpan={6} style={{ background: "var(--raised)" }}>
                  <div className="grid g2"><div><b className="small">Inputs</b><dl className="kv small" style={{ marginTop: 6 }}>{Object.entries(p.inputs).map(([k, v]: any) => <Fragment key={k}><dt>{k}</dt><dd>{String(v)}</dd></Fragment>)}</dl></div>
                    <div><b className="small">Top contributions</b><dl className="kv small" style={{ marginTop: 6 }}>{p.top_features.map((t: any) => <Fragment key={t.feature}><dt>{t.feature}</dt><dd className="num">{t.contribution > 0 ? "+" : ""}{t.contribution}</dd></Fragment>)}</dl>
                      <button className="btn sm" style={{ marginTop: 12 }} onClick={() => { setPatient(p.inputs); nav("/predict"); }}>Load into prediction studio</button></div></div></td></tr>}
              </Fragment>))}</tbody></table></div>
          <div className="row spread" style={{ marginTop: 14 }}><span className="muted small">Page {page + 1} of {pages}</span>
            <div className="row"><button className="btn sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><button className="btn sm" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button></div></div>
        </>)}
    </Panel>
  );
}

function Experiments() {
  const [kind, setKind] = useState(""); const [page, setPage] = useState(0); const [open, setOpen] = useState<number | null>(null);
  const q = `limit=${PAGE}&offset=${page * PAGE}${kind ? `&kind=${kind}` : ""}`;
  const { data, error, loading, reload } = useApi(() => api(`/history/experiments?${q}`), [q]);
  const del = async (id: number) => { await api(`/history/experiments/${id}`, { method: "DELETE" }); reload(); };
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1;
  return (
    <Panel title={`Experiments${data ? ` (${data.total})` : ""}`} aside={<div style={{ minWidth: 200 }}><select aria-label="Experiment kind" value={kind} onChange={(e) => { setKind(e.target.value); setPage(0); }}><option value="">All kinds</option><option value="training">Training runs</option><option value="counterfactual">Counterfactual searches</option><option value="scenario">What-if comparisons</option></select></div>}>
      {loading && !data ? <Loading /> : error ? <ErrorNote error={error} retry={reload} /> : data!.total === 0 ? <Empty title="No experiments">Training runs, counterfactual searches and what-if comparisons are recorded here.</Empty> : (
        <>
          {data!.items.map((e: any) => (
            <div key={e.id} style={{ borderBottom: "1px solid var(--line-soft)", padding: "12px 0" }}>
              <div className="row spread"><div><span className="pill neutral">{e.kind}</span> <b style={{ marginLeft: 8 }}>{e.title}</b><div className="faint small">{fmtDate(e.created_at)}{e.model_version ? ` · ${e.model_version}` : ""}</div></div>
                <div className="row"><button className="btn sm" onClick={() => setOpen(open === e.id ? null : e.id)} aria-expanded={open === e.id}>{open === e.id ? "Hide" : "Details"}</button><button className="btn danger sm" onClick={() => del(e.id)} aria-label={`Delete experiment ${e.id}`}>Delete</button></div></div>
              {open === e.id && <pre style={{ background: "var(--raised)", padding: 12, borderRadius: 8, overflowX: "auto", fontSize: ".8rem", marginTop: 10, maxHeight: 340 }}>{JSON.stringify({ summary: e.summary, config: e.config }, null, 2)}</pre>}
            </div>))}
          <div className="row spread" style={{ marginTop: 14 }}><span className="muted small">Page {page + 1} of {pages}</span><div className="row"><button className="btn sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><button className="btn sm" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button></div></div>
        </>)}
    </Panel>
  );
}

function Privacy() {
  const { data, error, loading } = useApi(() => api("/privacy"), []);
  if (loading) return <Loading />; if (error || !data) return <ErrorNote error={error} />;
  return <Panel title="What CardioSage stores and how to delete it"><dl className="kv">
    <dt>Location</dt><dd>{data.stored_locally_in}</dd><dt>Stored per prediction</dt><dd>{data.prediction_fields.join(", ")}</dd><dt>Never stored</dt><dd>{data.not_stored.join(", ")}</dd><dt>Deleting</dt><dd>{data.deletion}</dd></dl>
    <p className="muted small" style={{ marginTop: 12 }}>Counterfactual and what-if experiments also store the submitted feature values. Nothing is sent to any external service: the backend only serves this computer.</p></Panel>;
}

// Typed fetch layer. Every number shown in the UI comes from one of these calls; there are no mocked responses.
import { useCallback, useEffect, useRef, useState } from "react";

export class ApiError extends Error {
  status: number; code?: string; fields: Record<string, string>;
  constructor(status: number, message: string, code?: string, fields: Record<string, string> = {}) {
    super(message); this.status = status; this.code = code; this.fields = fields;
  }
}

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api${path}`, {
      credentials: "include",
      method: opts.method ?? (opts.body !== undefined ? "POST" : "GET"),
      headers: opts.body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    });
  } catch {
    throw new ApiError(0, "Cannot reach the CardioSage backend. Make sure it is running (run.ps1 starts it) and reload.", "network");
  }
  const text = await res.text();
  let json: any = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* non-JSON */ }
  if (!res.ok) {
    const d = json?.detail;
    if (Array.isArray(d)) {
      const fields: Record<string, string> = {};
      d.forEach((e: any) => { const k = String(e.loc?.[e.loc.length - 1] ?? "input"); fields[k] = e.msg; });
      throw new ApiError(res.status, "Some inputs are invalid: " + Object.entries(fields).map(([k, v]) => `${k}: ${v}`).join("; "), "validation", fields);
    }
    if (d && typeof d === "object") throw new ApiError(res.status, d.message ?? "Request failed", d.code);
    throw new ApiError(res.status, typeof d === "string" ? d : `Request failed (${res.status})`);
  }
  return json as T;
}

export function useApi<T = any>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const seq = useRef(0);
  const run = useCallback(() => {
    const id = ++seq.current;
    setLoading(true);
    fn().then((d) => { if (id === seq.current) { setData(d); setError(null); } })
      .catch((e) => { if (id === seq.current) setError(e instanceof ApiError ? e : new ApiError(0, String(e))); })
      .finally(() => { if (id === seq.current) setLoading(false); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(() => { run(); }, [run]);
  return { data, error, loading, reload: run };
}

/** Run an action on demand (button click) and track its state. */
export function useAction<A extends any[], T>(fn: (...a: A) => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async (...a: A) => {
    setLoading(true); setError(null);
    try { const r = await fn(...a); setData(r); return r; }
    catch (e) { setError(e instanceof ApiError ? e : new ApiError(0, String(e))); return null; }
    finally { setLoading(false); }
  };
  return { data, error, loading, run, reset: () => { setData(null); setError(null); } };
}

export const pct = (x: number | null | undefined, d = 1) => (x == null ? "â€”" : `${(x * 100).toFixed(d)}%`);
export const f3 = (x: number | null | undefined) => (x == null ? "â€”" : x.toFixed(3));
export const fmtDate = (s: string) => new Date(s).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

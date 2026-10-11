// Integration tests: the real React pages talk to a REAL running CardioSage backend (no mocks).
// Start the backend first (run.ps1 does this) or set VITE_TEST_API. Tests are skipped, with a clear message, if it is down.
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "../App";
import { AppProvider } from "../context";

const API = (import.meta.env.VITE_TEST_API as string | undefined) ?? "http://127.0.0.1:8000";
let up = false;

// jsdom lacks ResizeObserver, which Recharts' ResponsiveContainer needs (test-environment shim only).
(globalThis as any).ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} };

// Route relative /api calls to the real backend (jsdom has no origin).
const realFetch = globalThis.fetch;
globalThis.fetch = ((input: any, init?: any) => realFetch(typeof input === "string" && input.startsWith("/api") ? API + input : input, init)) as typeof fetch;

afterEach(() => cleanup());

beforeAll(async () => {
  try { up = (await realFetch(`${API}/api/health`)).ok; } catch { up = false; }
  if (!up) console.warn(`Backend not reachable at ${API}; integration tests skipped.`);
});

const pages: [string, RegExp][] = [
  ["/", /distinct records after removing/],
  ["/dataset", /Dataset explorer/],
  ["/predict", /Patient record/],
  ["/counterfactual", /Search settings/],
  ["/what-if", /Scenario B/],
  ["/benchmark", /Cross-validated performance/],
  ["/explain", /Permutation importance/],
  ["/history", /What is stored|Predictions/],
  ["/methodology", /Reliability evidence/],
];

describe("pages render real backend data", () => {
  for (const [path, text] of pages) {
    it(`renders ${path}`, async (ctx) => {
      if (!up) return ctx.skip();
      render(<MemoryRouter initialEntries={[path]}><AppProvider><App /></AppProvider></MemoryRouter>);
      await waitFor(() => expect(screen.getAllByText(text).length).toBeGreaterThan(0), { timeout: 30000 });
      expect(screen.queryByText(/Cannot reach the CardioSage backend/)).toBeNull();
      cleanup();
    });
  }

  it("overview shows the real duplicate count from the dataset", async (ctx) => {
    if (!up) return ctx.skip();
    render(<MemoryRouter initialEntries={["/"]}><AppProvider><App /></AppProvider></MemoryRouter>);
    await waitFor(() => expect(screen.getByText(/723 exact duplicates/)).toBeTruthy(), { timeout: 30000 });
    cleanup();
  });
});

describe("API contract used by the pages", () => {
  it("predict -> counterfactual -> scenario -> history", async (ctx) => {
    if (!up) return ctx.skip();
    const post = (p: string, b: unknown) => realFetch(API + p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then((r) => r.json());
    const f = { age: 57, sex: 1, cp: 0, trestbps: 140, chol: 241, fbs: 0, restecg: 1, thalach: 123, exang: 1, oldpeak: 0.2, slope: 1, ca: 0, thal: 3 };
    const p = await post("/api/predict", { features: f, save: false });
    expect(p.probability_class1).toBeGreaterThan(0); expect(p.explanation.contributions).toHaveLength(13);
    const cf = await post("/api/counterfactual", { features: f, save: false });
    expect(cf.found).toBe(true); expect(cf.candidates[0].verified).toBe(true);
    const sc = await post("/api/scenarios/compare", { baseline: f, scenario: { ...f, thalach: 170 }, save: false });
    expect(sc.changed_features).toEqual(["thalach"]);
    const bad = await realFetch(API + "/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features: { ...f, age: 3 } }) });
    expect(bad.status).toBe(422);
  });
});

describe("user flows through real buttons", () => {
  const mount = (path: string) => render(<MemoryRouter initialEntries={[path]}><AppProvider><App /></AppProvider></MemoryRouter>);

  it("prediction studio: run prediction shows verdict, explanation and model agreement", async (ctx) => {
    if (!up) return ctx.skip();
    localStorage.clear(); mount("/predict");
    await waitFor(() => expect(screen.getByLabelText("Age (years)")).toBeTruthy(), { timeout: 30000 });
    fireEvent.click(screen.getByRole("button", { name: /Run prediction/ }));
    await waitFor(() => expect(screen.getByText(/Strongest contributing features/)).toBeTruthy(), { timeout: 60000 });
    expect(screen.getByText(/Do the models agree/)).toBeTruthy();
    expect(screen.getByText(/Model output, not a diagnosis/)).toBeTruthy();
    cleanup();
  });

  it("prediction studio: invalid input is blocked with a clear message", async (ctx) => {
    if (!up) return ctx.skip();
    localStorage.clear(); mount("/predict");
    const age = await screen.findByLabelText("Age (years)", {}, { timeout: 30000 });
    fireEvent.change(age, { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: /Run prediction/ }));
    await waitFor(() => expect(screen.getByText(/Must be between 18 and 100/)).toBeTruthy());
    cleanup();
  });

  it("counterfactuals: search returns verified options", async (ctx) => {
    if (!up) return ctx.skip();
    localStorage.clear(); mount("/counterfactual");
    const btn = await screen.findByRole("button", { name: /Find counterfactuals/ }, { timeout: 30000 });
    await waitFor(() => expect((btn as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(btn);
    await waitFor(() => expect(screen.getAllByText(/re-verified/).length).toBeGreaterThan(0), { timeout: 60000 });
    expect(screen.getByText(/Option 1/)).toBeTruthy();
    cleanup();
  });

  it("what-if: editing B and comparing shows changed inputs and sensitivity", async (ctx) => {
    if (!up) return ctx.skip();
    localStorage.clear(); mount("/what-if");
    const hr = await screen.findByLabelText("Maximum heart rate (bpm)", {}, { timeout: 30000 });
    fireEvent.change(hr, { target: { value: "170" } });
    fireEvent.click(screen.getByRole("button", { name: /Compare scenarios/ }));
    await waitFor(() => expect(screen.getByText(/Changed inputs/)).toBeTruthy(), { timeout: 60000 });
    expect(screen.getByText(/Sensitivity: Maximum heart rate/)).toBeTruthy();
    cleanup();
  });

  it("history: saved predictions are listed and can be deleted", async (ctx) => {
    if (!up) return ctx.skip();
    await realFetch(API + "/api/history/predictions", { method: "DELETE" });
    const f = { age: 50, sex: 1, cp: 2, trestbps: 120, chol: 200, fbs: 0, restecg: 0, thalach: 160, exang: 0, oldpeak: 0, slope: 2, ca: 0, thal: 2 };
    await realFetch(API + "/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features: f, explain: false }) });
    mount("/history");
    await waitFor(() => expect(screen.getByText(/Predictions \(1\)/)).toBeTruthy(), { timeout: 30000 });
    fireEvent.click(screen.getByRole("button", { name: /Delete prediction/ }));
    await waitFor(() => expect(screen.getByText(/No saved predictions/)).toBeTruthy(), { timeout: 30000 });
    cleanup();
  });
});

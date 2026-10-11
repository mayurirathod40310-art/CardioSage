import { createContext, useContext, useEffect, useMemo, useState, ReactNode } from "react";
import { api, ApiError } from "./api";

export type Patient = Record<string, number>;
export interface FeatureSpec {
  name: string; label: string; kind: "numeric" | "binary" | "categorical" | "ordinal"; unit: string | null;
  description: string; options: Record<string, string> | null; hard: [number, number] | null; step: number | null; locked: boolean;
}
export interface SchemaResp { features: FeatureSpec[]; observed_ranges: Record<string, any>; locked_by_default: string[]; class_labels: Record<string, string> }

const EXAMPLE: Patient = { age: 57, sex: 1, cp: 0, trestbps: 140, chol: 241, fbs: 0, restecg: 1, thalach: 123, exang: 1, oldpeak: 0.2, slope: 1, ca: 0, thal: 3 };

interface Ctx {
  schema: SchemaResp | null; schemaError: ApiError | null;
  patient: Patient; setPatient: (p: Patient) => void;
  scenarioB: Patient | null; setScenarioB: (p: Patient | null) => void;
  algorithm: string | null; setAlgorithm: (a: string | null) => void;
  models: { key: string; label: string; selected: boolean }[];
}
const C = createContext<Ctx>(null as unknown as Ctx);
export const useApp = () => useContext(C);

function stored<T>(k: string, d: T): T { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } }
function save(k: string, v: unknown) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } }

export function AppProvider({ children }: { children: ReactNode }) {
  const [schema, setSchema] = useState<SchemaResp | null>(null);
  const [schemaError, setSchemaError] = useState<ApiError | null>(null);
  const [patient, setPatientState] = useState<Patient>(() => stored("cs.patient", EXAMPLE));
  const [scenarioB, setB] = useState<Patient | null>(() => stored("cs.scenarioB", null));
  const [algorithm, setAlg] = useState<string | null>(() => stored("cs.algo", null));
  const [models, setModels] = useState<Ctx["models"]>([]);

  useEffect(() => {
    const loadModels = () => api("/models").then((m) => setModels(m.algorithms)).catch(() => setModels([]));
    api<SchemaResp>("/dataset/schema").then(setSchema).catch((e) => setSchemaError(e));
    loadModels();
    window.addEventListener("cardiosage:retrained", loadModels); // fired by the benchmarking page after training
    return () => window.removeEventListener("cardiosage:retrained", loadModels);
  }, []);

  const value = useMemo<Ctx>(() => ({
    schema, schemaError, patient, scenarioB, algorithm, models,
    setPatient: (p) => { setPatientState(p); save("cs.patient", p); },
    setScenarioB: (p) => { setB(p); save("cs.scenarioB", p); },
    setAlgorithm: (a) => { setAlg(a); save("cs.algo", a); },
  }), [schema, schemaError, patient, scenarioB, algorithm, models]);
  return <C.Provider value={value}>{children}</C.Provider>;
}

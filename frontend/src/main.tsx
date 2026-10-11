import "@fontsource-variable/schibsted-grotesk";
import "@fontsource-variable/newsreader";
import "./styles.css";
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AppProvider } from "./context";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode><BrowserRouter><AppProvider><App /></AppProvider></BrowserRouter></React.StrictMode>
);

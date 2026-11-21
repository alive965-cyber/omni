import type { Metadata } from "next";
import "./globals.css";
import { Shell } from "@/components/shell";

export const metadata: Metadata = {
  title: "OmniPulse MBS v2.0 — Cognitive & Dyadic Behavioral Telemetry Lab",
  description:
    "Observable, baseline-relative trimodal + rPPG behavioral telemetry, 100ms micro-sequence encoding, and 3-Tier Stratified RAG synthesis workspace.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-scroll-behavior="smooth">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}

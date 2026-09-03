import type { Metadata } from "next";
import { Bricolage_Grotesque, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

// Display face: characterful grotesque for headlines.
// Mono: everything that reads like data (confidences, BPM, keys, stages).
const display = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--font-display",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Decomposition — hear your song in pieces",
  description:
    "Upload a song. Get separated stems, detected instruments, BPM, key and an AI-written breakdown.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body
        className={`${display.variable} ${mono.variable} antialiased min-h-screen`}
        style={{ fontFamily: "var(--font-mono)" }}
      >
        {children}
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Awareness",
  description: "Personal Helper for career and awerness",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

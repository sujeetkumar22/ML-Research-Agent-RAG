import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ArXiv RAG | ML Research Assistant",
  description: "Ask questions over 40+ landmark ML/AI research papers — get cited, grounded answers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}

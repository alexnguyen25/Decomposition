import type { NextConfig } from "next";

// Local development runs the FastAPI backend for uploads; set BACKEND_ORIGIN to
// proxy /api/* there so the browser sees one origin (no CORS).
//
// Production (Vercel) sets no BACKEND_ORIGIN: the examples are static assets and
// /api/chat is a Next.js route handler, so there is nothing to proxy. Uploading
// your own track is a local-only feature — see README.
const backendOrigin = process.env.BACKEND_ORIGIN;

const nextConfig: NextConfig = {
  async rewrites() {
    if (!backendOrigin) return [];
    return [
      // /api/chat is served by this app in every environment; everything else
      // (upload, job polling, stem files) belongs to the Python backend.
      { source: "/api/analyze", destination: `${backendOrigin}/api/analyze` },
      { source: "/api/jobs/:path*", destination: `${backendOrigin}/api/jobs/:path*` },
      { source: "/api/files/:path*", destination: `${backendOrigin}/api/files/:path*` },
    ];
  },
};

export default nextConfig;

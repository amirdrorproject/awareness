/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return {
      // fallback only applies when no Next.js route (including our own
      // dynamic app/api/keystatic/[...params] route) already matched -
      // keeps Keystatic's local-mode read/write calls from being proxied
      // to the Python backend along with everything else under /api/*.
      fallback: [
        {
          source: "/api/:path*",
          destination:
            process.env.NODE_ENV === "development"
              ? "http://127.0.0.1:8000/api/:path*"
              : "/api/",
        },
      ],
    };
  },
};

export default nextConfig;

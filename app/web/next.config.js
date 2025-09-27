/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable strict mode for better error handling
  reactStrictMode: true,
  
  // Enable SWC minification for better performance
  swcMinify: true,
  
  // Configure image domains for medical images
  images: {
    domains: ['localhost', '127.0.0.1'],
    dangerouslyAllowSVG: true,
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },
  
  // API rewrites to proxy backend requests with timeout handling
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/api/:path*`,
      },
      {
        source: '/files/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/files/:path*`,
      },
    ];
  },
  
  // Server configuration for handling long requests
  experimental: {
    serverComponentsExternalPackages: [],
    proxyTimeout: 180000, // 3 minutes timeout for API proxy
  },
  
  // Optimize for development
  env: {
    BACKEND_TIMEOUT: '180000', // 3 minutes
  },
  
  // Headers for security
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
        ],
      },
    ];
  },
  
  // TypeScript configuration
  typescript: {
    // Type checking is done in a separate process during development
    ignoreBuildErrors: false,
  },
  
  // ESLint configuration
  eslint: {
    ignoreDuringBuilds: false,
  },
};

module.exports = nextConfig;
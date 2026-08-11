import type { NextRequest } from "next/server";

// v8.0-fix: 空字符串/空白 env 不应走 ?? fallback（?? 只对 null/undefined 生效）。
// 优先级：NEXT_PUBLIC_BACKEND_BASE_URL → DEER_FLOW_INTERNAL_GATEWAY_BASE_URL → nginx:80。
const BACKEND_BASE_URL = (
  process.env.NEXT_PUBLIC_BACKEND_BASE_URL?.trim() ||
  process.env.DEER_FLOW_INTERNAL_GATEWAY_BASE_URL?.trim() ||
  "http://nginx:80"
).replace(/\/+$/, "");

function buildBackendUrl(pathname: string) {
  return new URL(pathname, BACKEND_BASE_URL);
}

async function proxyRequest(request: NextRequest, pathname: string) {
  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("connection");
  headers.delete("content-length");

  const hasBody = !["GET", "HEAD"].includes(request.method);
  const response = await fetch(buildBackendUrl(pathname), {
    method: request.method,
    headers,
    body: hasBody ? await request.arrayBuffer() : undefined,
  });

  return new Response(await response.arrayBuffer(), {
    status: response.status,
    headers: response.headers,
  });
}

export async function GET(request: NextRequest) {
  return proxyRequest(request, "/api/memory");
}

export async function DELETE(request: NextRequest) {
  return proxyRequest(request, "/api/memory");
}

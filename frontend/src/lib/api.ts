const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    const message =
      typeof error.detail === "string"
        ? error.detail
        : Array.isArray(error.detail)
          ? error.detail.map((item: { msg?: string }) => item.msg || JSON.stringify(item)).join("; ")
          : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string, headers?: HeadersInit) => request<T>(path, { headers }),
  post: <T>(path: string, body?: unknown, headers?: HeadersInit) =>
    request<T>(path, {
      method: "POST",
      body: body instanceof FormData ? body : JSON.stringify(body ?? {}),
      headers,
    }),
  upload: <T>(path: string, form: FormData, headers?: HeadersInit) =>
    request<T>(path, { method: "POST", body: form, headers }),
  url: (path: string) => `${API_BASE}${path}`,
};

export const apiBase = API_BASE;

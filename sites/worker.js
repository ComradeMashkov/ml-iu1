const CONTENT_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ttf": "font/ttf",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

function extension(pathname) {
  const filename = pathname.slice(pathname.lastIndexOf("/") + 1);
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

function assetRequest(request, pathname) {
  const url = new URL(request.url);
  url.pathname = pathname;
  return new Request(url, request);
}

async function fetchAsset(request, env) {
  const url = new URL(request.url);
  const candidates = [];

  if (url.pathname === "/" || url.pathname.endsWith("/")) {
    candidates.push(`${url.pathname}index.html`);
  } else {
    candidates.push(url.pathname);
    if (!extension(url.pathname)) {
      candidates.push(`${url.pathname}.html`, `${url.pathname}/index.html`);
    }
  }

  for (const pathname of candidates) {
    const response = await env.ASSETS.fetch(assetRequest(request, pathname));
    if (response.status !== 404) {
      const headers = new Headers(response.headers);
      const contentType = CONTENT_TYPES[extension(pathname)];
      if (contentType && !headers.has("content-type")) {
        headers.set("content-type", contentType);
      }
      headers.set("x-content-type-options", "nosniff");
      headers.set("referrer-policy", "strict-origin-when-cross-origin");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }
  }

  return new Response("Страница не найдена", {
    status: 404,
    headers: { "content-type": "text/plain; charset=utf-8" },
  });
}

export default {
  fetch(request, env) {
    return fetchAsset(request, env);
  },
};

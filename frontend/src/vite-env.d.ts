/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Origin of the HoopVista API (e.g. `https://api.example.com`). Leave unset in
   * local dev to use the Vite `/api` proxy; required for static deploys.
   */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// Authoring reference: https://docs.railway.com/infrastructure-as-code/reference
// This file is declarative only. Do not apply it without a reviewed deployment.
import { defineRailway, github, preserve, project, service } from "railway/iac";

export default defineRailway(() => {
  const api = service("itinerary-api", {
    source: github("Ztang-Yit-Xiaang/weather-aware-travel-itinerary-optimization", {
      branch: "codex/deploy-latest-travel-site",
    }),
    // The root Dockerfile supplies the build; no Railpack build override.
    start: "python scripts/run_hosted_app.py",
    healthcheck: "/healthz",
    healthcheckTimeout: 60,
    replicas: 1,
    env: {
      WEB_CONCURRENCY: "1",
      PRODUCT_COPILOT_ADAPTER: "deterministic",
      HOSTED_ENABLE_PAID_OPENAI: "false",
      // Set these manually before first deployment; preserve existing values.
      HOSTED_API_ORIGIN: preserve(),
      HOSTED_FRONTEND_ORIGIN: preserve(),
      HOSTED_OWNER_ID: preserve(),
      DATABASE_URL: preserve(),
      SUPABASE_URL: preserve(),
      SUPABASE_PUBLISHABLE_KEY: preserve(),
      ORS_API_KEY: preserve(),
    },
  });

  // Supabase is external: no Railway database, volume, or other paid resources.
  return project("weather-aware-itinerary", { resources: [api] });
});

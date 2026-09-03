Type: grilling
Status: resolved (2026-09-03)

## Question

The reference-app's abuse guards (per-IP cooldown 120s, global daily cap
200, 15 MB upload max, 6 min duration max, 30 min result TTL) were tuned as
reasonable defaults for a local dev demo, not for real public traffic on a
free-tier host. Also unresolved: domain/hosting specifics.

Needs a real decision on:
- Do the current thresholds hold for production, or do they need
  re-tuning against Modal's free credit budget and expected recruiter
  traffic (bursty, low volume, but zero tolerance for the demo looking
  broken during a screening)?
- Vercel project name / subdomain vs. a custom domain (cost, whether a
  custom domain is worth it for a resume link).
- Whether the Modal $30/mo free-credit budget needs a hard circuit breaker
  beyond the existing daily cap, given this is meant to run unattended for
  weeks during application season.

## Resolution (2026-09-03)

Mostly moot: the public site runs no analysis, so upload limits and daily
caps do not apply in production. Vercel Hobby hard-caps at 100 GB/month
and pauses rather than billing. The abuse guards remain in the backend for
local use. No custom domain for now.

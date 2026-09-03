Type: grilling
Status: resolved (2026-09-03)

## Question

The build ladder has both an HF Space (step 3, quick to ship, free, resume-
linkable URL) and a full Vercel+Modal web app (step 5, now confirmed in
scope per the application-ready destination, chat agent included). Once
the full web app exists, what's the Space for?

Needs a real decision on:
- Keep both permanently (Space as a lightweight fallback demo that never
  goes to sleep/cold-starts, full app as the flagship) vs. retire the Space
  once the full app is live vs. Space stays the *only* public demo and the
  full web app is a "see it running" GIF in the README instead of a second
  live deployment.
- If both stay: does the Space get the chat agent too, or does it stay the
  simpler one-shot-description version it already has (this affects whether
  ticket 03's provider choice needs to support two deployment targets)?
- Which one is "the" link that goes on the resume/LinkedIn/repo About field
  — recruiters skim ~7 seconds, so this should be one unambiguous URL, not
  two competing ones.

## Resolution (2026-09-03)

Moot: **HF Docker Spaces now require PRO ($9/mo)**. One Vercel project
serves everything — examples as static assets, chat as a Next.js route.
`reference-app/space/` kept as a record and marked obsolete.

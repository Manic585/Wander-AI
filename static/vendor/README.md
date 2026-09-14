# Response rendering dependencies

These browser builds are served locally so response formatting does not depend on a CDN.

- Marked 18.0.13: `marked/lib/marked.umd.js` from npm; license in `marked.LICENSE`.
- DOMPurify 3.4.15: `dompurify/dist/purify.min.js` from npm; license in `dompurify.LICENSE`.

Update the pinned versions and browser builds together, then verify Markdown rendering and HTML sanitization.

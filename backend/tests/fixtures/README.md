# Test fixtures

`gdelt_artlist.json` is **hand-written** to match the documented GDELT DOC 2.0 `mode=artlist&format=json`
response shape (keys: url, url_mobile, title, seendate, socialimage, domain, language, sourcecountry).
It is not a recorded response: GDELT returned HTTP 429 from the development machine while this was
written, so the live format has not been verified yet. Re-record it from a live call when possible
(wait a few minutes between calls; the limiter is stricter than the documented 1 request / 5 s).

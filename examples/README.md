# Examples

Each example is a Jev-compatible request that can be POSTed to `/v1/evaluate`
or piped through `python -m z_jev.infer_cli --checkpoint <path> --request <file>`.

* `spam_request.json` -- single Choice + Score + Noul over a synthetic SMS.
* `support_ticket_request.json` -- the canonical "support triage" example
  from the Jev spec (billing vs technical vs sales / urgency / frustration).

`sample_output.json` is the canonical model output captured by
`scripts/smoke.sh` and committed verbatim.
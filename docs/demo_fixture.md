# Demo Fixture — Brand Genome Engine

A stable, non-copyrighted sample genome for live demos and reproducible
screenshots. Paste these values into the Genome Setup form.

## Brand Designation
`Solenne Timepieces`

## Mission / Core Vision
`Solenne designs quiet, considered timepieces for people who value substance over spectacle. We believe a watch should age like a well-kept promise: precise, honest, and built to be handed down.`

## Exactly 7 snippets

1. `Every Solenne case is finished by hand over three days, because a rushed edge never feels right on the wrist.`
2. `We publish the sourcing story of every material we use, from the steel to the strap.`
3. `A Solenne watch is warrantied for life, not because we expect it to fail, but because we expect you to keep it.`
4. `Our design studio removes a feature for every one it adds; restraint is the hardest engineering problem we solve.`
5. `We test movements for four years before they ever reach a customer's wrist.`
6. `Solenne does not chase trends; a watch that looks dated in five years was never well designed.`
7. `Repairability is a promise: any Solenne watch, of any age, can be serviced by us.`

## Consistency Check example

Input text: `Solenne watches are the hottest new drop this season — grab yours before they sell out!!!`

Expect a **low** consistency score (casual, hype-driven tone vs the
measured/understated genome) with diagnostics flagging tone/vocabulary
mismatch.

## Benchmark example

Competitor: `rolex` (or any of the other 9 checked-in competitors). Metric
set: tone, sentiment, readability (canonical 3-metric contract).

## Rewrite example

Input text (same off-brand example above) run through `POST /api/rewrite`
(or the Rewrite page) should retrieve grounding snippets from the 7 above,
and produce a calmer, more restrained rewrite with a higher post-score than
pre-score (not guaranteed every run, but expected given the tone gap).

# QDS · Qutrit Competition Lab · Ultra

Single-file, offline **qutrit arena** (0 / 1 / 2) with:

- Touch-first simulation canvas (phone safe)
- Same parameter stack as original Qutrit Lab:
  - Grid size, speed, seed
  - Init 0 / P1 / P2 fractions
  - Competition temperature
  - Flip strength + symmetric flip noise
  - Lock support / margin / base probability / gain per margin
  - Toroidal wrap + optional rare unlock
- Live metrics:
  - Fraction (0 / P1 / P2)
  - Potential agreement (0 = chaos, 1 = locked)
  - Mean local cluster size
- **Quick fairness check**: 20 seeds × 120 steps, reporting mean |P1−P2| and mean P1/P2.

> Neon fairness sandbox · **not advice**  
> Local-only · **no external calls**

## Run locally

Just open the HTML file in a browser:

- Direct file: `qutrit_competition_lab_ultra.html`
- Or serve it from a static folder, e.g.:

```bash
python -m http.server 8011
# then open http://127.0.0.1:8011/qutrit_competition_lab_ultra.html

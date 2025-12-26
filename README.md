# QDS · Qutrit Competition Lab · Ultra

Neon, touch-first sandbox for **3-state competition**:

> 0 = Empty · 1 = P1 · 2 = P2

- **Live lab:**  
  https://danfromdursley-spec.github.io/qds-qutrit-competition-lab-ultra/qutrit_competition_lab_ultra.html  

---

## What this is

Single-file, offline **qutrit arena**:

- Touch-first simulation canvas (phone-safe)
- Same parameter stack as the original Qutrit Lab:
  - Grid size, speed, seed
  - Init 0 / P1 / P2 fractions
  - Competition temperature
  - Flip strength + symmetric flip noise
  - Lock support / margin / base probability / gain per margin
  - Toroidal wrap + optional rare unlock

All logic runs locally in the browser.  
No frameworks, no API calls, no tracking.

---

## Live metrics

- **Fractions (0 / P1 / P2)**
- **Potential agreement** (0 = chaos, 1 = locked)
- **Mean local cluster** size
- **Quick fairness check**: 20 seeds × 120 steps  
  → reports mean |P1–P2| and mean P1 / P2.

> Neon fairness sandbox · **not advice**  
> Local-only · **no external calls**

---

## Run locally

Just open the HTML file in a browser:

- Direct file: `qutrit_competition_lab_ultra.html`
- Or serve it from a static folder, e.g.:

```bash
python -m http.server 8011
# then open:
#   http://127.0.0.1:8011/qutrit_competition_lab_ultra.html

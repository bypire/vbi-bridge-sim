# Technical note: time-step restrictions for damped moving-load beams

A short, self-contained note built on this project's beam finite-element core.
It quantifies why an explicit Runge-Kutta scheme needs a tiny time step on a
damped bridge (the stiffness-proportional damping term, not the load speed, sets
the limit) and verifies two standard remedies against the Fryba closed form:
implicit Newmark-beta and modal truncation.

## Build
- Figures and numbers: `python note/make_note_figures.py`
  (writes `figures/` and `data.json`).
- PDF: `tectonic note.tex` (or `pdflatex note.tex`, run twice).

Every number in the note is reproduced by the script and checked against the
Fryba reference. AI tools were used to help prepare the code, figures and text;
the author has verified the results.

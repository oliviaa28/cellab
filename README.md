# CelLab

Sistem pentru automate celulare: utilizatorul descrie un automat (lumea ca matrice, tipurile de celule,
regulile de evolutie), sistemul il valideaza, construieste starea initiala, simuleaza N generatii,
le afiseaza si le salveaza, iar o simulare salvata poate fi reluata exact de unde a ramas.

Automatele si experimentele sunt descrise in fisiere JSON (formatul 2.0), nu in cod.
Codul nu contine nimic specific unui anumit automat.

## Instalare

Necesar: Python 3.12.

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` fixeaza versiunea NumPy: cu alta versiune, acelasi `seed` poate da alte numere aleatoare,
deci instalati exact versiunea de acolo.

## Utilizare

Toate comenzile se ruleaza din folderul proiectului (cel cu `demo.py`).

Rularea unui experiment:

```powershell
python demo.py                                         # Forest Fire, starea initiala din enunt
python demo.py experiments/rule_30_center.json
python demo.py experiments/brians_brain_random.json
```

Demo-ul:
1. valideaza experimentul si automatul (toate erorile sunt afisate; la eroare se opreste);
2. afiseaza automatul si parametrii experimentului;
3. construieste starea initiala si simuleaza generatiile;
4. afiseaza fiecare generatie salvata (numarul de celule pe stare si grila, daca e mica);
5. salveaza rularea completa in `runs/<nume>_<data>_<nr>/`;
6. face testul de recuperare: N pasi + salvare + reincarcare + M pasi == N + M pasi dintr-o data;
7. deseneaza generatiile (salvate si in `evolution.png` din folderul rularii).

Reluarea unei rulari salvate (continua cu acelasi generator aleator, deci rezultatul e identic
cu o rulare neintrerupta):

```powershell
python demo.py --resume runs/<folderul rularii> 10
```

Teste:

```powershell
python -m pytest
```

## Ce contine o rulare salvata

| Fisier | Continut |
|---|---|
| `automaton.json` | copia automatului |
| `experiment.json` | copia experimentului |
| `generations.npz` | generatiile pastrate (dupa `storage`) |
| `last_state.npz` | ultima stare (mereu) |
| `meta.json` | versiunea formatului, data, ultima generatie, generatiile pastrate, starea completa a generatorului aleator |

## Structura proiectului

```
automata/       automatele (JSON 2.0)
experiments/    experimentele (JSON 2.0)
schema/         schemele JSON pentru automat si experiment
cellab/         pachetul Python
  conventions.py      conventii comune (ordinea vecinilor, radacina proiectului)
  datatypes.py        forma datelor dupa incarcare
  validator.py        verificarea fisierelor (forma + sens)
  loader.py           incarcarea fisierelor valide
  engine_cells.py     motorul de reguli (vectorizat, NumPy)
  reference_engine.py motorul de referinta (celula cu celula), pentru verificare
  initial_state.py    starea initiala
  simulation.py       simularea (pas cu pas / N generatii)
  storage.py          salvare si reincarcare
  random_automata.py  automate aleatoare (teste si set de date)
tests/          teste; tests/invalid/ = fisiere gresite intentionat, cu eroarea asteptata
runs/           rulari salvate (nu intra in git)
```

Un fisier trece prin trei pasi: **forma** (JSON Schema) → **sens** (validator) → **incarcare** (loader).
Fiecare eroare are un cod (de ex. `UNKNOWN_STATE`), locul exact din fisier (JSON Pointer) si un mesaj.

## Formatul 2.0, pe scurt

Automat:

```json
{
  "format_version": "2.0",
  "kind": "cells",
  "name": "Forest Fire",
  "states": [
    {"name": "EMPTY", "symbol": ".", "color": "#8B6F47"},
    {"name": "TREE",  "symbol": "T", "color": "#2E7D32"},
    {"name": "FIRE",  "symbol": "F", "color": "#E4572E"}
  ],
  "neighborhood": {"type": "moore", "radius": 1},
  "rules": [
    {"from": "TREE", "to": "FIRE",  "when": {"count": "FIRE", "min": 1}},
    {"from": "FIRE", "to": "EMPTY"}
  ]
}
```

- `neighborhood.type`: `moore` (8 vecini), `von_neumann` (4), `1d` (stanga, dreapta).
- Regulile se verifica in ordine; pentru fiecare celula se aplica **prima** regula potrivita.
  Daca are `probability` si tragerea nu reuseste, se incearca regula urmatoare. Fara regula potrivita,
  celula ramane neschimbata. Toate celulele se calculeaza din generatia anterioara (sincron).
- `from`: o stare, o lista de stari sau `"*"` (orice stare).
- Conditii (`when`):
  - `{"count": "FIRE", "min": 1}`, `{"count": "ALIVE", "in": [3, 6]}` — numarul de vecini intr-o stare;
  - `{"count": ["I1", "I2", "I3"], "min": 2}` — numarul de vecini aflati in oricare din stari;
  - `{"pattern": ["ON", "OFF", "OFF"]}` — doar 1D: [stanga, celula, dreapta];
  - `{"all": [...]}`, `{"any": [...]}`, `{"not": ...}` — combinatii.
- `parameters`: valori numerice cu nume, folosite in `min`, `max`, `in`, `probability`;
  un experiment le poate suprascrie.

Experiment:

```json
{
  "format_version": "2.0",
  "automaton": "automata/forest_fire_stochastic.json",
  "parameters": {"p": 0.05},
  "world": {"rows": 60, "cols": 60, "boundary": "toroidal"},
  "initial_state": {"cells": {"mode": "random", "proportions": {"TREE": 0.6, "EMPTY": 0.4}}},
  "generations": 200,
  "storage": {"mode": "every", "every": 10},
  "seed": 7
}
```

- `boundary`: `toroidal` (marginile se unesc) sau `fixed` (cu `outside`: starea din afara grilei).
- Starea initiala: `fill`, `random` (proportii care dau 1), `manual` (grila din simboluri),
  `pattern` (un model pe un fundal, la o pozitie).
- `storage`: `all`, `last` sau `every` (din k in k generatii).
- `seed` e obligatoriu daca ceva e aleator; aceeasi valoare da mereu aceeasi simulare.

## Automate incluse

| Automat | Fisier | Experiment | Stare |
|---|---|---|---|
| Rule 30 | `rule_30.json` | `rule_30_center.json` | functioneaza |
| Brian's Brain | `brians_brain.json` | `brians_brain_random.json` | functioneaza |
| Cyclic | `cyclic.json` | `cyclic_random.json` | functioneaza |
| Forest Fire | `forest_fire.json` | `forest_fire_statement.json`, `forest_fire_manual.json` | functioneaza |
| Forest Fire probabilist | `forest_fire_stochastic.json` | `forest_fire_stochastic_random.json` | functioneaza |
| Traffic (Rule 184) | `traffic_184.json` | `traffic_184_road.json` | functioneaza |
| Epidemie determinista | `epidemic.json` | `epidemic_center.json` | functioneaza |
| Crystal Growth | `crystal_growth.json` | `crystal_growth_center.json`, `crystal_growth_at_least_one.json` | functioneaza |
| Voter | `voter.json` | — | faza 2 (formule) |
| Sandpile | `sandpile.json` | `sandpile_manual.json` | faza 2 (valori numerice) |
| Diffusion | `diffusion.json` | `diffusion_point.json` | faza 2 (valori numerice) |
| Langton's Ant | `langtons_ant.json` | `langtons_ant_center.json` | faza 3 (agenti) |
| Falling Sand | `falling_sand.json` | — | faza 3 (agenti) |

Automatele din faza 2 si 3 se valideaza si se incarca deja; demo-ul spune clar ca simularea lor
nu e inca disponibila, in loc sa ruleze gresit.

## Limite asumate

Grile 1D sau 2D patrate, timp discret, vecinatate de raza 1, regulile depind doar de celula si vecinii ei
din generatia anterioara. Nu sunt suportate: grile hexagonale sau grafuri, raza > 1, reguli care
depind de generatii mai vechi, agenti care ocupa mai multe celule.

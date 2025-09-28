# Cycling Power & Pacing Calculator

Minimal viable webová aplikace v Pythonu/Flask pro výpočet doporučeného výkonu a rychlosti na základě GPX trasy.

## Instalace
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Spuštění
```bash
python app.py
```
Aplikace poběží na http://localhost:5000.

## Režimy výpočtu
- **Cílová rychlost** – uživatel zadá cílovou rychlost (km/h) a aplikace spočítá, jaký výkon je potřeba udržet na jednotlivých segmentech (výchozí segment 200 m).
- **Cílové RPE** – uživatel zadá subjektivní úroveň námahy (RPE 1–10), která se přepočítá na procento FTP a výsledný výkon. Aplikace následně najde dosažitelnou rychlost pro každý segment.

## Výchozí fyzikální konstanty
- Hustota vzduchu `rho = 1.225 kg/m³`
- Součinitel odporu * čelní plocha `CdA = 0.30 m²`
- Koeficient valivého odporu `Crr = 0.004`

## Poznámka
Mapování RPE → %FTP je jen orientační. Hodnoty lze upravit v souboru `app.py` v sekci konstant.

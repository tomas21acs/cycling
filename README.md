# Cycling Power & Training Tools

Jednoduchá Flask aplikace, která nabízí dvě hlavní funkce:

1. **Kalkulačka výkonu / pacingu** – výpočet výkonu a rychlosti podle GPX trasy.
2. **Analýza tréninku** – vizualizace a souhrny z TCX/FIT souborů.

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

## Kalkulačka výkonu
- **Cílová rychlost** – uživatel zadá cílovou rychlost (km/h) a aplikace spočítá, jaký výkon je potřeba udržet na jednotlivých segmentech (výchozí segment 200 m).
- **Cílové RPE** – uživatel zadá subjektivní úroveň námahy (RPE 1–10), která se přepočítá na procento FTP a výsledný výkon. Aplikace následně najde dosažitelnou rychlost pro každý segment pomocí numerického řešení.
- Výstup zahrnuje tabulku segmentů, souhrnné statistiky, grafy výkonu/rychlosti/sklonu a možnost exportu CSV.

## Analýza tréninku
- Nahrajte TCX nebo FIT soubor – aplikace automaticky rozpozná formát.
- Výstup obsahuje mapu trasy (Leaflet), souhrnné statistiky (délka, čas, převýšení, průměrné/max hodnoty) a grafy pro tep, výkon, rychlost a nadmořskou výšku (Chart.js).
- Pokud data v souboru chybí (např. bez výkonu), aplikace daný graf vynechá a zobrazí informativní hlášku.

## Výchozí fyzikální konstanty
- Hustota vzduchu `rho = 1.225 kg/m³`
- Součinitel odporu * čelní plocha `CdA = 0.30 m²`
- Koeficient valivého odporu `Crr = 0.004`

## Poznámky
- Mapování RPE → %FTP je orientační. Hodnoty lze upravit v souboru `app.py` v sekci konstant.
- Analytické funkce využívají knihovnu `fitparse` pro FIT soubory a jednoduché zpracování TCX přes `xml.etree.ElementTree`.

# Cycling Power & Training Tools

Flask aplikace pro cyklisty, která kombinuje plánování výkonu i detailní analýzu reálných jízd.

1. **Kalkulačka výkonu / pacingu** – výpočet výkonu a rychlosti podle GPX trasy.
2. **Analýza tréninku** – výpočet pokročilých metrik z TCX/FIT a uložení do historie uživatele.

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

## Analýza tréninku & historie
- Po přihlášení lze nahrát TCX nebo FIT soubor (volitelně zadat FTP a název). Backend spočítá základní statistiky, pokročilé metriky (Normalized Power, IF, TSS, VI, kalorie) a uloží výsledek do SQLite databáze.
- Detaily tréninku zobrazují mapu trasy (Leaflet), souhrnné tabulky a čtyři grafy (Chart.js) – tep, výkon, rychlost a nadmořskou výšku.
- Každý uživatel má vlastní historii jízd s přehledovou tabulkou (datum, vzdálenost, TSS, NP, IF) a odkazem na detail.
- Pokud soubor neobsahuje potřebné datové kanály (např. GPS), aplikace uživatele informuje a trénink se neuloží.

## Uživatelské účty
- Registrace a přihlášení (Flask-Login + Flask-WTF, hesla hashována přes bcrypt).
- Každý uživatel vidí pouze své uložené tréninky.
- Po registraci je potřeba vyplnit stránku **Profil** (FTP, max. tep, váha) a přidat alespoň jedno kolo; tyto údaje se používají při analýze a volbě kola pro každý trénink.
- Nahrané soubory jsou uloženy v `instance/uploads/` a slouží k opětovnému zobrazení detailu.

## Výchozí fyzikální konstanty
- Hustota vzduchu `rho = 1.225 kg/m³`
- Součinitel odporu * čelní plocha `CdA = 0.30 m²`
- Koeficient valivého odporu `Crr = 0.004`

## Poznámky
- Mapování RPE → %FTP je orientační a lze jej upravit v souboru `app.py` v sekci konstant.
- Pro korektní běh analýzy FIT souborů je nutné mít nainstalovánu závislost `fitparse` (viz `requirements.txt`).

# 🧬 Forensics xG | Crime Scene Investigation
**Where the Beautiful Game Meets Hard Evidence**

Forensics xG is a commercial-grade, interactive football analytics dashboard built with Python and Streamlit. Designed for deep tactical deconstruction, it transforms raw Opta/Stats Perform JSON event data and tabular statistical reports into highly detailed, interactive visual evidence.

## 🔍 Overview
This dashboard acts as a digital laboratory for analyzing manager tactical profiles and player performances. It bypasses basic box-score stats to measure *how* and *where* a team controls the pitch, utilizing advanced geospatial mapping and proprietary mathematical engines.

## ✨ Key Features & Tactical Modules

### 📐 The Mathematical Engines
* **Universal Spatial Normalization:** Automatically standardizes coordinate data so the selected team is always attacking right (x=100) and the opponent is mapped attacking left (x=0), eliminating the "double-negative" mapping bugs common in raw event data.
* **Expected Threat (xT) Generator:** A custom 12x8 probability matrix evaluates every successful pass and carry, assigning a literal "On-Ball Value" based on how much the action increased the likelihood of scoring. 
* **Set-Piece Look-Ahead Logic:** Traces match flow dynamically, looking ahead up to 15 events to identify and highlight the exact corner or free kick that served as the primary catalyst for a goal.

### 📊 Statistical Reports
Interactive scatter plots plotting squad and player performance metrics (Standard, Advanced GK, Shooting, Goal/Shot Creation, Passing, Defense, and Possession) with dynamic per-90 normalization.

### ⚽ Match Telemetry Layers (Pitch Maps & Charts)
Over 35 stackable visual modules categorized by phase of play:
* **⚔️ Offensive:** High-Value xT Actions, Expected Threat Grid, Shot Creation Vectors, Fast Break Transitions, Attacking Lanes (5-Lane Theory), Progressive Carries, and Player Impact Boards (Net xT).
* **🛡️ Defensive:** Defensive Shields (Heatmaps & Avg Recovery Line), Zonal Defensive Pressure, Convex Hull Impact Zones, Defensive Penetrations Conceded, and Matchup positioning (Offense vs. Defense).
* **⚽ Possession:** Pass Networks, Pass Sonars (Distribution Radars), Zone 14 & Half-Space Invasions, Switches of Play, Momentum Maps, and Rolling 5-minute Possession control.
* **🎯 Set-Pieces:** Corner Deliveries and Free Kick Targeting with goal-creating catalysts highlighted.

## 🛠️ Installation & Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/jczpineda/Forensics-xG.git](https://github.com/jczpineda/Forensics-xG.git)
cd Forensics-xG

**2. Install dependencies:**
Bash
pip install -r requirements.txt

**3. Run the application locally:**
Bash
streamlit run forensics_app.py

🕵️‍♂️ About the Author
Created and maintained by Carlos Pineda.
Currently honing advanced analytical methodologies in the Sports Analytics program at Escuela Universitaria Real Madrid.

For more deep-dive tactical breakdowns and visual evidence, check out The Box-to-Box Investigator on YouTube.

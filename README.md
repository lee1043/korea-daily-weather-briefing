# AI Weather Intelligence Tools

Small AI-powered utilities for **meteorological analysis and weather information synthesis**.  
All workflows are operated automatically via GitHub Actions.  

---

## 1. Worldwide Extreme Weather / Climate News (search_news.py)

Automatically searches for **major global weather and climate events** and generates a concise summary.

**Features**
- AI-assisted global weather news search
- Selects major events from the last 24 hours
- Generates short meteorological summaries
- Extracts real source links
- Markdown output suitable for **Discord or PDF**

**Use Cases**
- Climate monitoring
- Daily extreme weather digest

![discord](discord.png)

---

## 2. AI Weather Briefing Generator (daily_briefing.py)

Generates an **automated synoptic weather briefing for the Korea Peninsula** using satellite imagery and NWP charts.

**Features**
- Downloads operational charts from **KMA**
- Multimodal AI analysis of:
  - GK2A satellite imagery
  - Surface analysis
  - 500 hPa geopotential height
  - 850 hPa wind
- Generates structured forecast sections:
  - Synoptic overview
  - 24–48h forecast features
  - Regional weather
  - Hazard assessment
  - Forecast uncertainty
- Produces a **PDF briefing report**
- Optional **Discord delivery**

![sample](sample.png)

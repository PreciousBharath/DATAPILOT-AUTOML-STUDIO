<div align="center">

# 🧪 DataPilot — AutoML & EDA Studio

**Drop in any CSV → instant visual EDA → one-click AutoML that trains, cross-validates and ranks 5 models → live predictions.**

![Python-3.10%2B](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Flask-3.x](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white) ![SQLite-Database](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white) ![Chart.js-Charts](https://img.shields.io/badge/Chart.js-Charts-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white) ![Pandas-EDA](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge&logo=pandas&logoColor=white) ![scikit--learn-AutoML](https://img.shields.io/badge/scikit--learn-AutoML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

`Data Science · AutoML` · `Full-stack web app` · `Register / Login` · `Dark & light themes`

</div>

---

![DATAPILOT-AUTOML-STUDIO](assets/datasets.png)

## ✨ Features

- **Drag & drop CSV upload** (or 3 built-in sample datasets)
- **Automatic EDA**: types, missing values, duplicates, skew, IQR outliers, distributions for every column
- Interactive **correlation heatmap** and **auto-generated insights**
- **One-click AutoML** — auto task detection, imputation, scaling, one-hot encoding, 5 models, 3-fold CV leaderboard
- Feature importance, **confusion matrix** or **actual-vs-predicted** plot
- **Live prediction form** generated from the dataset schema; experiments saved & reloadable

## 🎨 UI / UX highlights

- **Glassmorphism design system** with animated gradient blobs, grid overlay and per-project colour theme
- **Dark / light mode** toggle (remembered per browser) — every chart re-themes instantly
- **Split-screen auth pages** with an interactive particle network, floating cards and a typing "Try demo" button
- **Motion everywhere**: staggered fade-ups, count-up numbers, animated progress bars, ripple buttons, toast notifications, modals
- **Fully responsive** — collapsible sidebar on mobile, Ctrl + K search shortcut
- Works **offline**: Chart.js, Bootstrap Icons and Poppins font are bundled in `static/`

## 🔐 Authentication & security

- **Register / Login / Logout** with salted password hashing (Werkzeug PBKDF2)
- Live **password-strength meter**, show/hide password, "remember me" sessions (14 days)
- **CSRF protection** on every form and API call, brute-force lockout after 5 failed logins
- HttpOnly + SameSite cookies, security headers, per-user data isolation
- **Profile & settings**: edit name, avatar colour, change password, delete account

## 🖼️ Screenshots

| Sign in | Register |
|---|---|
| ![login](assets/login.png) | ![register](assets/register.png) |

| | |
|---|---|
| ![eda](assets/eda.png) | ![automl](assets/automl.png) |
| ![automl_class](assets/automl_class.png) | ![experiments](assets/experiments.png) |

## 🏗️ How it works

```mermaid
flowchart LR
    S0["Upload CSV"] --> S1
    S1["profile & visualise"] --> S2
    S2["choose target"] --> S3
    S3["train & rank models"] --> S4
    S4["inspect"] --> S5
    S5["predict"]
```

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask, SQLite, pandas, numpy, scikit-learn |
| Frontend | HTML5, CSS3 (custom glassmorphism design system), vanilla JavaScript |
| Charts & icons | Chart.js 4, Bootstrap Icons, Poppins (bundled locally) |
| Security | Werkzeug password hashing, CSRF tokens, secure sessions |

## 📁 Project structure

```
DATAPILOT-AUTOML-STUDIO/
├── app.py                 # Flask app: routes, business logic & JSON API
├── core.py                # Shared core: app factory, SQLite, auth, CSRF, profile
├── app.py                 # Routes, uploads, experiments, prediction API
├── engine.py              # Profiling + AutoML engine & sample datasets
├── templates/             # Jinja2 pages (auth, base layout, feature pages)
├── static/                # style.css, main.js + bundled Chart.js, icons & fonts
├── assets/                # README screenshots
├── requirements.txt
└── LICENSE
```

## 🚀 Run it locally

```bash
git clone https://github.com/PreciousBharath/DATAPILOT-AUTOML-STUDIO.git
cd DATAPILOT-AUTOML-STUDIO
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** → click **"Try the demo account"** (`demo@example.com` / `Demo@1234`) or register your own account.
The SQLite database is created automatically in `instance/` and seeded with realistic demo data.

## ⚙️ Production tips

- Set a strong `SECRET_KEY` environment variable and run behind **gunicorn** / **waitress** with HTTPS
- `DATABASE=/path/to/app.db` overrides the database location

## 👨‍💻 Author

**Bharath** — Software Developer · Project Engineer · Technical Trainer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-connectbharath-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/connectbharath)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://itsbharathportfolio.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-PreciousBharath-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PreciousBharath)

⭐ If you like this project, please give it a star!

## 📄 License

Released under the [MIT License](LICENSE).

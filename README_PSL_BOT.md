# PSL-бот (looksmaxxing-разбор по фото)

## Запуск за 5 минут
1. `@BotFather` в ТГ → `/newbot` → скопируй токен.
2. https://aistudio.google.com/app/apikey → создай бесплатный `GEMINI_API_KEY`.
3. В корне проекта:
   ```bash
   copy .env.example .env
   ```
   вставь токены в `.env`.
4. Установка и старт:
   ```bash
   pip install -r requirements.txt
   python bot.py
   ```
5. Кинь боту фото лица → получишь замеры + жёсткий разбор.

## Как это работает
- **MediaPipe FaceLandmarker** (локально, 478 точек, модель ~3.6MB
  докачивается сама при первом запуске во временную папку) считает:
  fWHR, jaw/zygo, mouth/zygo, lower-third, canthal tilt (hunter/prey),
  асимметрию.
- Цифры + фото уходят в **Gemini** с системным промптом
  «хирург + consensual aesthetic analysis» → структурированный вердикт:
  PSL-тир, failo/halo, anti-fraud, softmaxxing-план, прогноз ascend.
- Фото не хранится: временный файл удаляется сразу после анализа.

## Команды
- `/start` — приветствие
- `/help` — как фоткаться чтобы разбор был точным
- фото в чат — анализ

## Деплой на Render (бесплатно, 24/7, без карты)
Почему Render: из РФ закрыт Gemini API, а зарубежный сервер решает сразу
две задачи — бот живёт 24/7 и Gemini отвечает без VPN.

1. Залей на GitHub файлы: `bot.py`, `requirements-render.txt`
   (переименуй в `requirements.txt` в репозитории ИЛИ укажи
   `pip install -r requirements-render.txt` в Build command).
   MediaPipe в облако НЕ ставим — free-тариф 512MB RAM, бот сам
   перейдёт в лайт-режим (разбор только по фото, без замеров).
2. render.com → Sign up через GitHub → New → Web Service → выбери репо.
   - Build command: `pip install -r requirements.txt`
   - Start command: `python bot.py`
   - Instance type: Free
3. Environment → переменные:
   `TELEGRAM_TOKEN`, `GEMINI_API_KEY`, `WEBHOOK_SECRET` (любая абракадабра).
   `RENDER_EXTERNAL_URL` и `PORT` Render подставит сам.
4. Deploy → в Logs жди `webhook-режим: https://...onrender.com` → бот живёт.
5. ⚠️ Выключи локальный `python bot.py` — два экземпляра дерутся (ошибка 409).
6. Анти-сон: на cron-job.org заведи GET на `https://твой-сервис.onrender.com/`
   каждые 10 минут (корень отдаёт `ok`).

Локально ничего менять не надо: без `RENDER_EXTERNAL_URL` бот сам
стартует в polling-режиме с полной биометрией.

## Деплой на Hugging Face Spaces (⛔ МЁРТВ для бесплатников с лета 2026)
HF требует PRO ($9/мес) для создания ЛЮБЫХ compute-Spaces (Docker и Gradio).
Бесплатно — только Static (сайты, не боты). Раздел ниже оставлен для истории.
/ В архиве: Dockerfile пригодится для будущего VPS.

## Деплой на Hugging Face Spaces (было актуально до paywall)
Почему HF: бесплатно, без карты, 16GB RAM на free CPU — влезает
**полная версия с MediaPipe-биометрией** (на Render только лайт).
Спячка после 48ч тишины лечится тем же cron-job.org.

1. huggingface.co → Sign up → New Space:
   - имя любое (например `psl-bot`), SDK: **Docker**, Hardware: CPU basic (free),
   - Visibility: Public (приватный Space тоже ок, боту всё равно).
2. В файлы Space залей ТОЛЬКО 4 файла (не всю папку! там гигабайты мусора):
   `bot.py`, `requirements.txt` (**полный, с mediapipe**), `Dockerfile`,
   `.dockerignore`.
3. Settings → Variables and secrets → добавь Secrets (не Variables!):
   `TELEGRAM_TOKEN`, `GEMINI_API_KEY`, `WEBHOOK_SECRET` (абракадабра),
   `PORT=7860`, `PUBLIC_URL=https://ТВОЙ-НИК-psl-bot.hf.space`
   (точный URL виден сверху на странице Space).
4. Space сам пересоберётся (Factory rebuild, ~5 минут) → в Logs ищи
   `webhook-режим: https://...hf.space` → бот живёт, с полными замерами.
5. ⚠️ Выключи бота везде где он ещё крутится (комп, Render) — иначе 409.
6. Анти-сон: cron-job.org → GET на `https://...hf.space/` каждые 30 минут.

## Почему НЕ Netlify
Serverless-функции с лимитом ~10 сек на ответ и без постоянного процесса:
наш разбор (биометрия + Gemini) занимает 20-60 сек и не влезет в лимиты,
а mediapipe не влезет в лимит размера бандла. Netlify — для сайтов, не для ботов.

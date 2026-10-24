# FROSS — настройка публикации

## 1. Firebase и вход через Google или email/пароль
1. Открой https://console.firebase.google.com/ и создай проект Firebase.
2. В настройках проекта добавь Web App. Скопируй объект конфигурации приложения.
3. В Firebase Console → Authentication → Sign-in method включи Google и Email/Password (Email/Password — именно пароль, не email link). Phone/SMS не включай.
4. В Authentication → Settings → Authorized domains добавь домен `fross.boberbuset132013.workers.dev`.
5. В Firebase Console → Firestore Database создай базу данных.
6. Открой `public/firebase-config.js` в GitHub и замени все значения `YOUR_...` на значения из Firebase Web App. Это публичная конфигурация клиента; не вставляй сюда service-account JSON или приватный ключ.
7. В Firestore → Rules вставь содержимое файла `firestore.rules` и опубликуй правила.

При первом входе через Google аккаунт создаётся автоматически. Через email пользователь может зарегистрироваться, придумать пароль, войти позже и запросить письмо для сброса пароля. SMS/номер телефона не используются.

## 2. FROSS 1.1 с Brain 0.5: распределённая схема
Сайт остаётся на JavaScript, но может передавать запросы в отдельные сервисы:
- обычный режим → `FROSS Python Brain 0.5` (`services/python-brain/server.py`, использует `fross_brain_v05.py`);
- режим «Рассуждение» → `FROSS C++ Reasoning Brain 0.5` (`services/cpp-brain/main.cpp`, HTTP-обёртка находится в `services/cpp-brain/server.py`);
- если сервис не настроен, недоступен или не понимает запрос, Worker использует локальный JavaScript-движок и при необходимости поиск.

Это собственные алгоритмы, правила, вычислитель и шаблоны — не готовая языковая модель. C++-ответы включают более подробный разбор задачи, но это не скрытое рассуждение обученной нейросети.

## 3. Подключение Python и C++ сервисов бесплатно
Проекты сервисов уже находятся в репозитории; файл `render.yaml` описывает два Docker-сервиса с планом Free. Render Free может приостанавливать неактивные сервисы, поэтому первый запрос после перерыва может быть медленнее или перейти на резервный JavaScript-ответ. Подробности: [официальные ограничения Render Free](https://render.com/docs/free).

1. Открой [Render](https://render.com/) и создай Blueprint, выбрав репозиторий `boberbuset132013-web/FROSS`. Render должен прочитать файл `render.yaml` и предложить `fross-python-brain` и `fross-cpp-brain`. Выбирай **Free** у обоих; не выбирай платный план.
2. Создай один секретный токен в PowerShell:
   ```powershell
   [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
   ```
   Вставь одинаковое значение в переменную `FROSS_BRAIN_SERVICE_TOKEN` для обоих Render-сервисов. Токен нельзя публиковать в GitHub, сообщениях чата или коде.
3. Дождись успешной сборки двух сервисов. В Render открой каждый сервис и скопируй его реальный публичный URL. Проверь, что адреса `/health` открываются и показывают версию 0.5.
4. В Cloudflare Dashboard открой **Workers & Pages → fross → Settings → Variables and Secrets**. Добавь текстовые переменные:
   - `FROSS_PYTHON_BRAIN_URL` = URL Python-сервиса с путём `/chat`, например `https://ИМЯ-PYTHON-СЕРВИСА.onrender.com/chat`;
   - `FROSS_CPP_BRAIN_URL` = URL C++-сервиса с путём `/chat`, например `https://ИМЯ-CPP-СЕРВИСА.onrender.com/chat`.
   
   Вместо примеров вставь адреса, которые реально показал Render.
5. Там же добавь **Secret** с именем `FROSS_BRAIN_SERVICE_TOKEN` и вставь точно такой же токен, как в обоих Render-сервисах. Не сохраняй его в `wrangler.jsonc` или других публичных файлах. Сохрани переменные и выполни новый deploy Worker, если Dashboard не сделал это автоматически.
6. После этого обычные запросы идут в Python, а глубокие — в C++. Если один из сервисов отвечает ошибкой или не поддерживает задачу, Worker попробует локальную JavaScript-логику и нужный поиск, вместо того чтобы полностью ломать чат.

Официальная документация Render: [Blueprints](https://render.com/docs/blueprint-spec), [Free plan и ограничения](https://render.com/docs/free).

При включённых внешних движках текст нового сообщения и последние сообщения текущего чата могут быть переданы в Render-сервис Python или C++, вместе с выбранными заметками памяти. Сервисы настроены не записывать содержимое сообщений в журналы, но всё равно не отправляй в чат пароли, токены, закрытые ключи или другие секреты.

## 4. Пароль создателя, память и GitHub-база знаний
1. В Cloudflare Dashboard открой **Workers & Pages → fross → Settings → Variables and Secrets → Add → Secret**. Имя секрета — `FROSS_CREATOR_PASSWORD`; в значение введи новый уникальный пароль длиной не менее 24 символов. Не добавляй пароль в `vars`, GitHub, HTML, JavaScript, `wrangler.jsonc` или README. Если используешь Wrangler CLI, выполни `npx wrangler secret put FROSS_CREATOR_PASSWORD` и введи пароль в приглашении. [Документация Cloudflare Secrets](https://developers.cloudflare.com/workers/configuration/secrets/).
2. Для записи общих учебных материалов в GitHub создай **fine-grained personal access token** в GitHub → Settings → Developer settings → Fine-grained personal access tokens. Ограничь доступ только репозиторием `boberbuset132013-web/FROSS`; в Repository permissions установи **Contents: Read and write** (Metadata обычно требуется как Read-only). Для Contents API записи нужны права на чтение/запись содержимого репозитория: [официальная таблица прав GitHub](https://docs.github.com/en/rest/authentication/permissions-required-for-fine-grained-personal-access-tokens).
3. В Cloudflare → Workers & Pages → `fross` → Settings → Variables and Secrets → Add → Secret создай секрет `FROSS_GITHUB_TOKEN` и вставь в его значение выданный GitHub token. **Никогда не добавляй токен в публичный репозиторий, `vars`, HTML, сообщения или JSON.** Сохрани и выполни Deploy Worker. Если секрет не настроен, FROSS по-прежнему будет сохранять личные заметки в Firestore, но запись в GitHub будет выключена.
4. Войди в FROSS через Google или email/пароль. Открой **«Обучение FROSS»**, введи пароль создателя и нажми **«Стать создателем»**. Сервер выдаёт подписанную сессию на 12 часов, привязанную к Firebase-пользователю.
5. В Firebase → Firestore Database → Rules опубликуй актуальный файл `firestore.rules`. Личные записи находятся в `users/{uid}/knowledge` и доступны только соответствующему вошедшему аккаунту.
6. Введи тему, например **«Основы Python»**, и нажми **«Изучить»**. FROSS собирает источники, извлекает краткое резюме, тезисы и ключевые слова, а затем:
   - сохраняет личные заметки в Firestore;
   - при настроенном `FROSS_GITHUB_TOKEN` записывает учебный материал в публичную папку `memory/knowledge/` и обновляет `memory/knowledge/index.json`.
7. В последующих ответах FROSS ищет подходящие темы в общей GitHub-базе и комбинирует их с личными заметками Firestore и свежими результатами поиска. В интерфейсе есть ссылка **«Открыть базу GitHub»**. Вкладка GitHub — публичная, а Firestore остаётся личным хранилищем аккаунта.

**Важно о публичной памяти:** весь контент в `memory/knowledge/` виден всем читателям репозитория и остаётся в истории Git. Храни там только общие учебные сведения, выдержки из открытых источников, резюме, тезисы и ссылки. Не сохраняй личные переписки, имена/почту пользователя, пароли, секреты, токены или другие приватные данные.

Это память с извлечением знаний, а не изменение весов нейросети. Резюме создаётся выбором предложений из найденного текста и может быть неполным, поэтому перепроверяй важные утверждения по ссылкам.
## 5. Публикация и проверка API
В Cloudflare Worker должно быть подключено это GitHub-репозитория. Команда развёртывания: `npx wrangler deploy`. Конфигурация `wrangler.jsonc` публикует папку `public` как статические файлы и передаёт запросы `/api/*` в Worker. После изменения репозитория дождись автоматического деплоя и обнови сайт сочетанием Ctrl+F5.

После публикации открой `https://fross.boberbuset132013.workers.dev/api/health`. Актуальная сборка должна вернуть JSON с `"version":"1.1"` и `"brainVersion":"0.5"`. Поля `creatorPasswordConfigured`, `githubKnowledgeConfigured`, `pythonBrainConfigured` и `cppBrainConfigured` показывают только факт настройки, не раскрывая значения секретов. Если вместо JSON приходит HTML, проверь маршрут/последнюю публикацию. Если API возвращает JSON с `errorId` и `diagnostic`, открой Cloudflare → Workers & Pages → `fross` → **Logs / Observability → Real-time Logs** и найди ошибку с этим ID: ответ API теперь логирует stack trace, а в чат отдаёт краткую диагностику. Никогда не отправляй в чат или логи значения секретов.

## 6. Возможности
- Вход/регистрация через Google и email/пароль.
- Восстановление пароля по email.
- Отдельные чаты с историей, сохранённой в Firestore.
- До 60 сообщений пользователя в каждом чате; на 60-м сообщение чат закрывается и становится доступен только для чтения. Можно создать новый чат.
- Анимация ответа «думает / ищет / формулирует».
- FROSS Brain 0.5 использует собственные правила, коррекцию частых опечаток, безопасный вычислитель арифметики, расширенную локальную базу понятий и сводки веб-поиска; готовая языковая модель не вызывается.
- Входящий API-запрос проверяет Firebase ID token и владельца чата.

## 7. Удаление чатов
В боковой панели рядом с каждым чатом есть кнопка `×`. Она удаляет сообщения этого чата и сам документ чата. Для этого нужно опубликовать обновлённые правила из `firestore.rules`: удалять документы может только вошедший владелец соответствующего пользователя. Старые чаты и их история будут удалены без возможности восстановления.

## 8. Если не работает
- Ошибка конфигурации/кнопка Google выключена: проверь `public/firebase-config.js`.
- `Unexpected token '<'` или сообщение `вернул HTML вместо JSON`: API получил HTML-страницу вместо JSON. Обычно сайт обслуживается старой/не той публикацией или маршруты `/api/*` не попадают в Worker. Проверь `wrangler.jsonc` (в `assets.run_worker_first` должен быть `"/api/*"`), сделай Deploy актуального Worker и обнови сайт через Ctrl+F5. В обычном чате запрос должен идти на `/api/chat`; секрет создателя сам по себе эту ошибку не вызывает.
- Не получается стать создателем: сначала войди в FROSS и открой `https://fross.boberbuset132013.workers.dev/api/health`. Если `creatorPasswordConfigured` равно `false`, в Cloudflare → Workers & Pages → `fross` → Settings → Variables and Secrets добавь именно **Secret** `FROSS_CREATOR_PASSWORD` длиной не менее 24 символов, сохрани и опубликуй новую версию Worker. Затем обнови сайт через Ctrl+F5 и введи это же значение в окне «Обучение FROSS». Если `creatorPasswordConfigured` равно `true`, проверь, что вводишь точное значение секрета без лишних пробелов. Альтернативная настройка `FROSS_CREATOR_EMAIL` работает только для совпадающего адреса с подтверждённой почтой Firebase. `FROSS_GITHUB_TOKEN` отвечает за запись общей базы знаний и сам по себе не даёт права создателя.
- `auth/unauthorized-domain`: добавь домен Worker в Firebase Authentication → Authorized domains.
- `Missing or insufficient permissions`: проверь опубликованные `firestore.rules`.
- Ответы слишком шаблонные: это ограничение rule-based FROSS Local Brain; расширяй `fross-brain.js` новыми правилами и примерами.
- После изменений дождись завершения автоматического деплоя и обнови страницу сочетанием Ctrl+F5.


## 9. Приватное распознавание создателя (настраивается отдельно)
Чтобы FROSS узнавал авторизованный аккаунт создателя, не добавляй адрес электронной почты в исходный код или публичные файлы GitHub. В Cloudflare Dashboard открой Workers & Pages → выбери Worker `fross` → Settings → Variables and Secrets → Add → Secret. Имя секрета: `FROSS_CREATOR_EMAIL`. В значение вставь личный адрес создателя, затем сохрани секрет и выполни новый deploy, если Dashboard его запросит.

Worker сравнивает секрет с email в проверенном Firebase ID token; адрес должен быть подтверждён Firebase (`email_verified = true`). Подтверждённый Google-вход обычно подходит. Если используешь email/пароль, аккаунт должен пройти подтверждение почты. Сам email не передаётся движку ответов и не хранится в публичном репозитории. Сервер использует его только для проверки флага создателя. Для связи с автором указывается только публичный Telegram.

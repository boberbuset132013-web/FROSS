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

## 2. Workers AI
В `wrangler.jsonc` уже добавлен binding Workers AI с именем `AI`. При деплое Cloudflare должен связать Worker с Workers AI. Если Dashboard попросит включить Workers AI, включи его в настройках Cloudflare. Модель `@cf/meta/llama-3.3-70b-instruct-fp8-fast` формирует ответы; её использование подчиняется квотам и условиям тарификации Cloudflare.

## 3. Публикация
В Cloudflare Worker должно быть подключено это GitHub-репозитория. Команда развёртывания: `npx wrangler deploy`. Конфигурация `wrangler.jsonc` публикует папку `public` как статические файлы и передаёт запросы `/api/*` в Worker. После изменения репозитория дождись автоматического деплоя и открой https://fross.boberbuset132013.workers.dev.

## 4. Возможности
- Вход/регистрация через Google и email/пароль.
- Восстановление пароля по email.
- Отдельные чаты с историей, сохранённой в Firestore.
- До 60 сообщений пользователя в каждом чате; на 60-м сообщение чат закрывается и становится доступен только для чтения. Можно создать новый чат.
- Анимация ответа «думает / ищет / формулирует».
- FROSS 1.0 отвечает через Workers AI и может добавлять найденные ссылки.
- Входящий API-запрос проверяет Firebase ID token и владельца чата.

## 5. Если не работает
- Ошибка конфигурации/кнопка Google выключена: проверь `public/firebase-config.js`.
- `auth/unauthorized-domain`: добавь домен Worker в Firebase Authentication → Authorized domains.
- `Missing or insufficient permissions`: проверь опубликованные `firestore.rules`.
- Ошибка Workers AI: проверь binding `AI` в Cloudflare и журнал Worker.
- После изменений подожди завершения автоматического деплоя и обнови страницу сочетанием Ctrl+F5.

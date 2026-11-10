# Публикация правил Firestore для FROSS

Firebase Authentication и Firestore — отдельные части проекта. Публикация коммита в GitHub или деплой сайта через Cloudflare сама по себе **не публикует** `firestore.rules` в Firebase.

Для FROSS используется Firebase-проект `fross-ca5d6` и стандартная база Firestore `(default)`. Файл `firebase.json` указывает Firebase CLI использовать корневой `firestore.rules`, а `.firebaserc` выбирает этот проект.

## Вариант 1 — через Firebase Console

1. Открой [Firestore Rules](https://console.firebase.google.com/project/fross-ca5d6/firestore/rules).
2. Убедись, что выбран проект `fross-ca5d6` и база `(default)`.
3. Сравни редактор с файлом [`firestore.rules`](https://github.com/boberbuset132013-web/FROSS/blob/main/firestore.rules).
4. Если содержимое отличается, замени его содержимым файла из репозитория и нажми **Publish**.
5. Подожди распространения правил и обнови FROSS. Firebase предупреждает, что новые запросы обычно подхватывают правила примерно за минуту, а активные слушатели могут обновляться дольше.

## Вариант 2 — Firebase CLI

Из корневой папки репозитория, где лежат `firebase.json` и `firestore.rules`, выполни:

```bash
npm install -g firebase-tools
firebase login
firebase deploy --only firestore:rules --project fross-ca5d6
```

Команда публикует только правила Firestore — она не меняет Google Sign-In, пользователей или документы чатов.

## Что защищает правило чатов

Для пути `/users/{userId}/chats/{chatId}` чтение разрешается только когда пользователь вошёл в Firebase Authentication и `request.auth.uid == userId`. Сообщения внутри `/messages/{messageId}` защищены тем же UID. Не заменяй это правилом `allow read, write: if true` — оно откроет чужие чаты.

Если после публикации всё ещё появляется `Missing or insufficient permissions`, проверь, что Google-вход в FROSS показывает тот же UID, что и владелец документов, и что правила опубликованы именно в проекте `fross-ca5d6`, в базе `(default)`.

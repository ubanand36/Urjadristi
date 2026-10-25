# UrjaDrishti Firebase setup

## Local development

```sh
npm install
npm run dev
```

The site uses Firebase Authentication email/password accounts. In the Firebase console for `urjadrishti-a7987`, open **Authentication → Sign-in method** and enable **Email/Password**. Add `localhost` under **Authentication → Settings → Authorized domains** if Firebase rejects local sign-in.

## Build and deploy

```sh
npm run build
firebase login
npm run deploy
```

The Hosting project alias is already set to `urjadrishti-a7987`. The Hosting target is `dist`, built by Vite. Review the generated build before deploying to the live site.

## Machine-learning API

The trained model and FastAPI setup is documented in [ML_PIPELINE.md](ML_PIPELINE.md). Firebase Hosting serves the frontend only; the Python inference service must be hosted separately. The API requires a valid Firebase Authentication ID token.

## Data and security scope

Firebase verifies account creation and sign-in. Machine readings, alerts, and access codes remain in browser storage in this prototype; they are not synced between accounts and the Settings/Reports codes are not server-enforced. Firebase Web App configuration identifies the client app and is intended for browser use. Never add a Firebase service-account JSON file or private key to this client project.

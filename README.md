# GymVision AI MVP

Local-first access-control demo for GymVision AI. The current MVP is a **demo-mode** application: it accepts synthetic 512-dimensional face embeddings, checks a submitted liveness score, consent, member status, and membership dates, then records an access decision. It does not capture or save face images.

## Important limits

- Demo mode stores its data in process memory. Restarting the application clears demo records.
- The MVP does not detect or align faces and does not include a selected or tested face-recognition or liveness model.
- Recognition is cosine similarity between supplied 512-dimensional vectors. Input vectors are not derived from a camera image.
- PostgreSQL 17 and pgvector are the approved production direction. This workspace currently has no verified PostgreSQL or Docker runtime, so PostgreSQL integration is not yet verified.
- This is a competition demo, not a production access-control or biometric system.

## Run locally

Python, FastAPI, Pydantic, Uvicorn, bcrypt, and pytest must already be installed. No dependencies are installed by this project automatically.

Copy `.env.example` to `.env` and replace both example credentials before running. The app reads this simple local file directly; it does not need a dotenv package. `GYMVISION_DATABASE_URL` and `GYMVISION_DEVICE_API_KEY` are reserved for the not-yet-integrated PostgreSQL/device path and are not consumed by this demo.

```powershell
python -m uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000` for the demo dashboard and `http://127.0.0.1:8000/docs` for the local API docs.

Authentication requires `GYMVISION_ADMIN_USERNAME` and `GYMVISION_ADMIN_PASSWORD`. The password is bcrypt-hashed in process memory at startup; only the hash is used to check login. Never use the example values outside a disposable local demo.

## Demo flow

1. Enroll a member with consent and a 512-number embedding.
2. Submit a check-in embedding and a liveness score from 0 to 1.
3. Review the allow/deny decision and access log.

The provided vectors and liveness score are demo inputs, not biometric evidence.

## Database

`database/schema.sql` remains the approved baseline. On a fresh database, the intended manual order is: `database/migrations/000_enable_vector.sql`, `database/schema.sql`, then `database/migrations/001_mvp_constraints.sql`. The two migration files contain only pre-approved additions: pgvector activation, cosine HNSW index, non-deleted phone/email unique indexes, selected NOT NULL constraints, a membership overlap trigger, and an append-only audit trigger. Apply nothing until a local PostgreSQL 17 + pgvector environment is available and the target is confirmed. The running MVP does not connect to PostgreSQL; no migration has been executed as part of this MVP work.

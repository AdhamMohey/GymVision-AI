# GymVision AI — Project Constitution

## 1. Project Vision

Build a local-first Smart Gym Access Control System powered by Computer Vision and Face Recognition.

Expected scale: approximately 5,000 gym members.

Core capabilities:

- Member enrollment.
- Face recognition.
- Face embedding extraction.
- Liveness detection.
- Membership validation.
- Access allow/deny decisions.
- Device management.
- Admin dashboard.
- Access and audit logging.

## 2. Approved Technology Stack

- Database: PostgreSQL 17.
- Vector extension: pgvector.
- Backend: Python and FastAPI.
- Containerization: Docker.
- Development: VS Code and Codex Desktop.

Do not replace or introduce major technologies without approval.

## 3. Approved Database Architecture

The approved database ERD is:

docs/diagrams/gymvision_erd_v1.png

This image is the authoritative reference for PostgreSQL ERD v1.0.

It contains nine tables:

1. members
2. membership_plans
3. memberships
4. face_embeddings
5. biometric_consents
6. devices
7. access_logs
8. admin_users
9. audit_logs

Approved baseline:

- UUID primary keys.
- VECTOR(512) embeddings.
- Cosine similarity.
- HNSW indexing.
- Local-first database architecture.
- Approximately 5,000 members.

## 4. ERD Preservation Rules

- Never modify, redraw, crop, replace, or delete the approved ERD image.
- Never independently change approved tables, columns, relationships, or data types.
- Do not add or remove tables without explicit approval.
- If a technical issue is discovered, explain it and propose a solution first.
- Keep implementation improvements separate from approved ERD changes.
- Always refer to the original ERD image when discussing database structure.

## 5. Security and Privacy

- Never store plaintext passwords.
- Never commit credentials or secrets.
- Use environment variables for sensitive configuration.
- Require valid biometric consent before enrollment and recognition.
- Enforce consent revocation.
- Protect biometric embeddings.
- Keep member data and biometric processing local.
- Do not upload sensitive member data to external services without approval.

## 6. Coding Standards

- Write clean, modular, maintainable code.
- Use Python type hints.
- Follow PEP 8.
- Handle errors explicitly.
- Validate user input.
- Write tests for important business logic.
- Avoid unnecessary dependencies.

## 7. Development Workflow

Follow this sequence:

1. Requirements and planning.
2. Architecture review.
3. Database specification.
4. Database implementation.
5. Backend development.
6. AI development.
7. System integration.
8. Dashboard development.
9. Testing.
10. Deployment.

Do not skip stages.

## 8. Mandatory Codex Behavior

- Inspect existing files before making changes.
- Explain the objective before implementation.
- Explain commands before asking the owner to execute them.
- Ask approval before destructive operations.
- Ask approval before database migrations or schema changes.
- Never modify the approved ERD independently.
- Never claim an operation succeeded without verification.
- Report changed files and test results.
- Do not install packages without approval.

## 9. Communication

The project owner is learning software engineering while building this project.

Therefore:

- Communicate in Egyptian Arabic.
- Keep technical terms in English where appropriate.
- Explain every unfamiliar concept.
- Explain why each tool or command is needed.
- Break tasks into small steps.
- Never ask the owner to execute unexplained commands.

## 10. Final Principle

Build GymVision AI as a professional, secure, maintainable software product.

Prioritize correctness, security, data integrity, and understanding over speed.

Preserve all approved architectural decisions unless the owner explicitly approves a change.

END OF PROJECT INSTRUCTIONS.

.PHONY: api web infra
api:
	cd backend && uvicorn app.main:app --reload --port 8000
web:
	cd frontend && npm run dev
infra:
	docker compose up -d

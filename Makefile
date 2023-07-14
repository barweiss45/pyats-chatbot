.PHONY: start
start:
	uvicorn main:app --reload --host 0.0.0.0 --port 9000

.PHONY: format
format:
	black .
	isort .
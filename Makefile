# ===== Makefile =====

# 启动 FastAPI 服务
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


train:
	python ml/train_v01.py


install:
	pip install -r requirements.txt

# 清理缓存文件
clean:
	rm -rf __pycache__ */__pycache__

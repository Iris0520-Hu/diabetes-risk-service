# ========= 基础镜像 =========
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 先复制依赖清单并安装（利用缓存）
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制代码与模型工件
COPY app/ app/
COPY artifacts/ artifacts/

# 暴露端口
EXPOSE 8000

# 健康检查（可选）
# HEALTHCHECK CMD curl -fsS http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]

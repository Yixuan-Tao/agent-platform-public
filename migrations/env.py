"""Additive migrations use configured database; no credentials in alembic.ini."""
import asyncio
import os
import sys
from pathlib import Path
from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

root = Path(__file__).resolve().parents[1]
for package in ("core", "orchestrator", "tools", "skills", "memory", "agents", "evaluation"):
    sys.path.insert(0, str(root / "packages" / package / "src"))
from agent_platform.core.db import Base
from agent_platform.orchestrator import runtime_store  # registers runtime tables
from agent_platform.core.config import get_settings

def migrate(connection):
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()

async def online():
    engine = create_async_engine(os.environ.get("MIGRATION_DATABASE_URL") or get_settings().database_url)
    async with engine.begin() as connection:
        await connection.run_sync(migrate)
    await engine.dispose()

if context.is_offline_mode():
    raise RuntimeError("本迁移先检查既有表；请使用在线迁移")
else:
    asyncio.run(online())

"""Create durable runtime tables, preserving all existing records and paths."""
from alembic import op
from sqlalchemy import inspect
import sqlalchemy as sa
from agent_platform.core.db import Base
from agent_platform.orchestrator import runtime_store

revision = "20261003_runtime"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # checkfirst supports both existing deployments and empty integration databases.
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)
    for table in ("feedback", "chat_messages"):
        columns = {c["name"] for c in inspect(op.get_bind()).get_columns(table)}
        if "run_id" not in columns:
            op.add_column(table, sa.Column("run_id", sa.String(64), nullable=True))

def downgrade():
    raise RuntimeError("回滚只停用新入口；禁止删除运行账本及历史数据")

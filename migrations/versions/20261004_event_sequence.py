"""Enforce replay sequence uniqueness without altering historical payloads."""
from alembic import op
from sqlalchemy import inspect

revision = "20261004_event_sequence"
down_revision = "20261003_runtime"
branch_labels = None
depends_on = None

def upgrade():
    connection = op.get_bind()
    inspector = inspect(connection)
    names = {row["name"] for row in inspector.get_unique_constraints("run_events")}
    names.update(row["name"] for row in inspector.get_indexes("run_events"))
    if "uq_run_event_sequence" not in names:
        op.create_index("uq_run_event_sequence", "run_events", ["run_id", "seq"], unique=True)

def downgrade():
    raise RuntimeError("回滚只停用入口，保留运行与审计记录")

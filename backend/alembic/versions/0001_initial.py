"""initial: documents + services

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-20
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("object_key", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("parent_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("kind", sa.String(16), nullable=False, server_default="source"),
        sa.Column("language_code", sa.String(8), nullable=True),
        sa.Column("service_name", sa.String(32), nullable=True),
        sa.Column("operation_name", sa.String(32), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.UniqueConstraint("object_key"),
    )
    op.create_index("ix_documents_id", "documents", ["id"])
    op.create_index("ix_documents_parent_id", "documents", ["parent_id"])

    op.create_table(
        "services",
        sa.Column("name", sa.String(32), primary_key=True),
        sa.Column("provider", sa.String(32), nullable=False, server_default="openai_compatible"),
        sa.Column("base_url", sa.String(512), nullable=False),
        sa.Column("api_key", sa.String(512), nullable=True),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("reasoning_effort", sa.String(16), nullable=True, server_default="low"),
        sa.Column("use_json_schema", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("limit_requests", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("max_input_tokens", sa.Integer(), nullable=True),
        sa.Column("chunk_tokens", sa.Integer(), nullable=True),
        sa.Column("max_output_tokens", sa.Integer(), nullable=True),
        sa.Column("languages", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("operations", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("input_token_cost", sa.Float(), nullable=True),
        sa.Column("output_token_cost", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("services")
    op.drop_index("ix_documents_parent_id", table_name="documents")
    op.drop_index("ix_documents_id", table_name="documents")
    op.drop_table("documents")

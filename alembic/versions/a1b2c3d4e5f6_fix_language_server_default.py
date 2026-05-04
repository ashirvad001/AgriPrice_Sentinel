"""fix_language_server_default

Revision ID: a1b2c3d4e5f6
Revises: ffe4ba3f60b3
Create Date: 2026-04-19 01:08:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'ffe4ba3f60b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add server_default='English' to the language column for existing deployments."""
    op.alter_column('alert_subscriptions', 'language',
        existing_type=sa.String(20),
        server_default='English',
        existing_nullable=False)


def downgrade() -> None:
    """Remove the server default from language column."""
    op.alter_column('alert_subscriptions', 'language',
        existing_type=sa.String(20),
        server_default=None,
        existing_nullable=False)

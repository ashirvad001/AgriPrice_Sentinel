"""add_rawprice_unique_constraint

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-19 01:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add unique constraint on (crop, state, fetch_date) to raw_prices.

    Before creating the constraint, deduplicate existing rows by keeping
    only the most recently inserted record for each (crop, state, fetch_date).
    """
    # Remove duplicates (keep highest id for each group)
    op.execute("""
        DELETE FROM raw_prices
        WHERE id NOT IN (
            SELECT MAX(id) FROM raw_prices
            GROUP BY crop, state, fetch_date
        )
    """)
    op.create_unique_constraint(
        'uq_rawprice_crop_state_date',
        'raw_prices',
        ['crop', 'state', 'fetch_date'],
    )


def downgrade() -> None:
    """Remove the unique constraint."""
    op.drop_constraint('uq_rawprice_crop_state_date', 'raw_prices', type_='unique')

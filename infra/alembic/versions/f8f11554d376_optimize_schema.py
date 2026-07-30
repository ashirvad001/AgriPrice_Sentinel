"""optimize_schema

Revision ID: f8f11554d376
Revises: b2c3d4e5f6a7
Create Date: 2026-07-26 20:50:25.303679

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8f11554d376'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Change raw_data column to JSONB (only on PostgreSQL)
    op.execute(
        "ALTER TABLE raw_prices ALTER COLUMN raw_data TYPE JSONB USING raw_data::JSONB"
    )
    
    # 2. Replace individual indexes with a composite index
    op.drop_index('ix_raw_prices_crop', table_name='raw_prices', if_exists=True)
    op.drop_index('ix_raw_prices_state', table_name='raw_prices', if_exists=True)
    op.drop_index('ix_raw_prices_fetch_date', table_name='raw_prices', if_exists=True)
    
    op.create_index(
        'idx_rawprice_crop_state_date', 
        'raw_prices', 
        ['crop', 'state', 'fetch_date'], 
        unique=False
    )


def downgrade() -> None:
    # 1. Revert index
    op.drop_index('idx_rawprice_crop_state_date', table_name='raw_prices', if_exists=True)
    
    op.create_index('ix_raw_prices_crop', 'raw_prices', ['crop'], unique=False)
    op.create_index('ix_raw_prices_state', 'raw_prices', ['state'], unique=False)
    op.create_index('ix_raw_prices_fetch_date', 'raw_prices', ['fetch_date'], unique=False)
    
    # 2. Revert JSONB back to JSON
    op.execute(
        "ALTER TABLE raw_prices ALTER COLUMN raw_data TYPE JSON USING raw_data::JSON"
    )

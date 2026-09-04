"""Alembic environment driven by ``akl-cli db``."""

from __future__ import annotations

from logging.config import dictConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from akl.db.models import Base

config = context.config

dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"plain": {"format": "%(levelname)s [%(name)s] %(message)s"}},
        "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
        "loggers": {"alembic": {"handlers": ["console"], "level": "INFO", "propagate": False}},
    }
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection")
    if connectable is None:
        engine = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                compare_server_default=True,
                transaction_per_migration=False,
            )
            with context.begin_transaction():
                context.run_migrations()
        engine.dispose()
    else:
        context.configure(
            connection=connectable, target_metadata=target_metadata, compare_type=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

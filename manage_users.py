#!/usr/bin/env python3
"""
CLI tool for managing users in Paperless AI Analyzer.

Usage (run inside the container):
    python3 /app/manage_users.py create --username alice --password secret --role admin
    python3 /app/manage_users.py list
    python3 /app/manage_users.py reset-password --username alice --password newpass
    python3 /app/manage_users.py deactivate --username alice
    python3 /app/manage_users.py activate --username alice

Run from host:
    docker exec -it paperless-ai-analyzer-dev python3 /app/manage_users.py create \\
        --username admin --password <pass> --role admin
"""

import sys
import argparse
import sqlite3
from pathlib import Path

# Ensure analyzer package is importable when running from /app
sys.path.insert(0, '/app')

from analyzer.db import init_db, create_user, get_user_by_username, list_users, update_user


def cmd_create(args):
    init_db()
    existing = get_user_by_username(args.username)
    if existing:
        print(f"ERROR: User '{args.username}' already exists.")
        sys.exit(1)
    create_user(
        username=args.username,
        password=args.password,
        role=args.role,
        display_name=args.display_name or args.username,
    )
    print(f"✓ Created user '{args.username}' with role '{args.role}'")


def cmd_list(args):
    init_db()
    rows = list_users()
    if not rows:
        print("No users found.")
        return
    fmt = "{:<5} {:<20} {:<25} {:<8} {:<20} {:<8}"
    print(fmt.format("ID", "Username", "Display Name", "Role", "Last Login", "Active"))
    print("-" * 92)
    for r in rows:
        print(fmt.format(
            r['id'],
            r['username'],
            r['display_name'] or '',
            r['role'],
            r['last_login'] or 'never',
            'yes' if r['is_active'] else 'no',
        ))


def cmd_reset_password(args):
    init_db()
    row = get_user_by_username(args.username)
    if not row:
        # Try inactive users
        import sqlite3 as _sq
        conn = _sq.connect('/app/data/app.db')
        conn.row_factory = _sq.Row
        row = conn.execute("SELECT * FROM users WHERE username = ?", (args.username,)).fetchone()
    if not row:
        print(f"ERROR: User '{args.username}' not found.")
        sys.exit(1)
    update_user(row['id'], password=args.password)
    print(f"✓ Password updated for '{args.username}'")


def cmd_deactivate(args):
    init_db()
    import sqlite3 as _sq
    conn = _sq.connect('/app/data/app.db')
    conn.row_factory = _sq.Row
    row = conn.execute("SELECT * FROM users WHERE username = ?", (args.username,)).fetchone()
    if not row:
        print(f"ERROR: User '{args.username}' not found.")
        sys.exit(1)
    update_user(row['id'], is_active=0)
    print(f"✓ User '{args.username}' deactivated")


def cmd_activate(args):
    init_db()
    import sqlite3 as _sq
    conn = _sq.connect('/app/data/app.db')
    conn.row_factory = _sq.Row
    row = conn.execute("SELECT * FROM users WHERE username = ?", (args.username,)).fetchone()
    if not row:
        print(f"ERROR: User '{args.username}' not found.")
        sys.exit(1)
    update_user(row['id'], is_active=1)
    print(f"✓ User '{args.username}' activated")


def main():
    parser = argparse.ArgumentParser(description='Manage Paperless AI Analyzer users')
    sub = parser.add_subparsers(dest='command', required=True)

    # create
    p_create = sub.add_parser('create', help='Create a new user')
    p_create.add_argument('--username', required=True)
    p_create.add_argument('--password', required=True)
    p_create.add_argument('--role', default='basic', choices=['basic', 'admin'])
    p_create.add_argument('--display-name', dest='display_name', default=None)

    # list
    sub.add_parser('list', help='List all users')

    # reset-password
    p_reset = sub.add_parser('reset-password', help='Reset a user password')
    p_reset.add_argument('--username', required=True)
    p_reset.add_argument('--password', required=True)

    # deactivate
    p_deact = sub.add_parser('deactivate', help='Deactivate (soft-delete) a user')
    p_deact.add_argument('--username', required=True)

    # activate
    p_act = sub.add_parser('activate', help='Re-activate a deactivated user')
    p_act.add_argument('--username', required=True)

    args = parser.parse_args()

    dispatch = {
        'create': cmd_create,
        'list': cmd_list,
        'reset-password': cmd_reset_password,
        'deactivate': cmd_deactivate,
        'activate': cmd_activate,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()

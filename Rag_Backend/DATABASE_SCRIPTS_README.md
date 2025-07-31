# Database Management Scripts

This directory contains scripts for managing the SixthVault PostgreSQL database.

## Scripts Overview

### 1. `reset_database.py` - Complete Database Reset
Drops all existing tables and recreates them from current models.

**⚠️ WARNING: This will delete ALL existing data!**

```bash
# Show help and safety warning
python reset_database.py

# Actually reset the database (requires confirmation)
python reset_database.py --confirm
```

**What it does:**
- Drops all existing tables in dependency order
- Creates new tables from current SQLModel definitions
- Verifies the new structure
- Runs basic connectivity tests
- Provides detailed logging of the process

### 2. `db_manager.py` - Comprehensive Database Management
Multi-purpose database management utility with various commands.

```bash
# Show all available commands
python db_manager.py

# Reset database (same as reset_database.py)
python db_manager.py reset --confirm

# Backup database to SQL file
python db_manager.py backup [filename]
python db_manager.py backup my_backup.sql

# Restore database from backup
python db_manager.py restore backup_file.sql

# Show database status and statistics
python db_manager.py status

# Clean up expired/orphaned records
python db_manager.py clean

# Seed database with initial data
python db_manager.py seed
```

## Database Structure

The scripts manage the following tables:

### User Management
- `users` - Main user accounts
- `temp_users` - Temporary users during registration
- `user_tokens` - Authentication tokens

### Document Management
- `document` - Uploaded documents
- `processing_documents` - Document processing status

### AI Features
- `ai_curations` - AI-generated document curations
- `curation_settings` - User curation preferences
- `document_curation_mapping` - Document-to-curation relationships
- `curation_generation_history` - Curation generation logs

### AI Summaries
- `ai_summaries` - AI-generated summaries
- `summary_settings` - User summary preferences
- `document_summary_mapping` - Document-to-summary relationships
- `summary_generation_history` - Summary generation logs

## Prerequisites

1. **PostgreSQL Client Tools** (for backup/restore):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql-client
   
   # macOS
   brew install postgresql
   
   # Windows
   # Download from https://www.postgresql.org/download/windows/
   ```

2. **Environment Variables** (in `.env` file):
   ```env
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_USER=your_user
   POSTGRES_PASSWORD=your_password
   POSTGRES_DB=your_database
   ```

3. **Python Dependencies**:
   ```bash
   pip install sqlmodel sqlalchemy psycopg2-binary
   ```

## Common Use Cases

### 1. Development Reset
When you need to start fresh during development:
```bash
python reset_database.py --confirm
```

### 2. Before Major Changes
Create a backup before making significant changes:
```bash
python db_manager.py backup before_changes.sql
# Make your changes
# If something goes wrong:
python db_manager.py restore before_changes.sql
```

### 3. Regular Maintenance
Clean up expired records and optimize:
```bash
python db_manager.py clean
```

### 4. Monitoring
Check database health and statistics:
```bash
python db_manager.py status
```

## Safety Features

- **Confirmation Required**: Destructive operations require explicit confirmation
- **Backup Verification**: Backup files are checked before restore
- **Dependency Handling**: Tables are dropped/created in proper order
- **Error Handling**: Comprehensive error reporting and rollback
- **Logging**: Detailed output for troubleshooting

## Troubleshooting

### Connection Issues
```bash
# Test database connection
python -c "from app.database import engine; print('Connection OK')"
```

### Permission Issues
Ensure your PostgreSQL user has the necessary permissions:
```sql
GRANT ALL PRIVILEGES ON DATABASE your_database TO your_user;
GRANT ALL ON SCHEMA public TO your_user;
```

### Missing Tables After Reset
If tables are missing after reset, check:
1. All models are properly imported in the scripts
2. SQLModel metadata includes all tables
3. No syntax errors in model definitions

### Backup/Restore Failures
- Ensure `pg_dump` and `psql` are in your PATH
- Check PostgreSQL client version compatibility
- Verify database connection parameters

## Best Practices

1. **Always backup before destructive operations**
2. **Test scripts in development environment first**
3. **Monitor database size and performance regularly**
4. **Clean up expired records periodically**
5. **Keep backups in a secure location**

## Integration with Application

These scripts work with the existing database configuration in `app/database.py` and use the same models defined in `app/models.py` and `app/database.py`.

The scripts automatically:
- Load configuration from `app/config.py`
- Import all necessary models
- Use the same database connection settings as your application

## Support

If you encounter issues:
1. Check the error messages in the script output
2. Verify your database connection settings
3. Ensure all dependencies are installed
4. Check PostgreSQL server logs for additional details

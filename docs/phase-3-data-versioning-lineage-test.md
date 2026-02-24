# NEO Hybrid AI — Data Versioning & Lineage Test & Integration

## Test Script (Python with DVC)
# Initialize DVC
# dvc init
# Track a dataset
# dvc add data/raw/historical_data.csv
# Commit changes
# git add . && git commit -m "Add versioned historical data"

## Test Script (Custom Metadata Table)
# Example: Insert lineage info into PostgreSQL
import psycopg2
conn = psycopg2.connect(dbname='neoai_db', user='neoai', password='neoai123', host='localhost', port=5432)
cur = conn.cursor()
cur.execute('''
    INSERT INTO data_lineage (dataset_name, version, source, processing_steps, created_at)
    VALUES (%s, %s, %s, %s, NOW())
''', ('historical_data.csv', 'v1', 'external API', 'normalized, feature engineered'))
conn.commit()
cur.close()
conn.close()

---
## Logging
- Log test results and integration steps.
- Update this file as integration evolves.
# NEO Hybrid AI — PostgreSQL Setup Documentation

## Container Details
- **Container Name:** neo-postgres
- **Image:** postgres:15
- **Port:** 5432
- **User:** neoai
- **Password:** neoai123
- **Database:** neoai_db

## Connection String
- `postgresql://neoai:neoai123@localhost:5432/neoai_db`

## Verification Steps
1. Container started via Docker.
2. Database neoai_db exists and is accessible.
3. Connection tested with `docker exec neo-postgres psql -U neoai -d neoai_db -c '\l'`.

## Next Steps
- Document schema for historical/log data.
- Integrate with Java/Python services.

---
This file documents the PostgreSQL setup for NEO Hybrid AI. Update as schema evolves.
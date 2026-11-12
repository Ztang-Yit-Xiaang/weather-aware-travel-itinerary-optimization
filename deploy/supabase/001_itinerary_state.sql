-- Apply explicitly as the backend database owner, once, before starting the app.
-- Keep itinerary_private OUT of Supabase's exposed Data API schemas.
-- Do not put the database credentials in GitHub Pages or use a browser role here.
-- The initial deployment supports one backend worker and one replica only.
BEGIN;

CREATE SCHEMA IF NOT EXISTS itinerary_private;
REVOKE ALL ON SCHEMA itinerary_private FROM PUBLIC;

CREATE TABLE IF NOT EXISTS itinerary_private.state_namespaces (
    namespace uuid PRIMARY KEY,
    revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0)
);

CREATE TABLE IF NOT EXISTS itinerary_private.state_files (
    namespace uuid NOT NULL REFERENCES itinerary_private.state_namespaces(namespace) ON DELETE CASCADE,
    path text NOT NULL CHECK (
        octet_length(path) BETWEEN 1 AND 512
        AND path ~ '^[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$'
        AND path ~ '\.(json|jsonl)$'
    ),
    content bytea NOT NULL CHECK (octet_length(content) <= 8388608),
    sha256 text NOT NULL CHECK (sha256 = encode(pg_catalog.sha256(content), 'hex')),
    PRIMARY KEY (namespace, path)
);

-- No browser policies: RLS denies access to non-owner roles by default.
ALTER TABLE itinerary_private.state_namespaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE itinerary_private.state_files ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON ALL TABLES IN SCHEMA itinerary_private FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA itinerary_private REVOKE ALL ON TABLES FROM PUBLIC;

-- Conditional role lookup also permits running this migration in an isolated
-- plain PostgreSQL test container, where Supabase's browser roles do not exist.
DO $migration$
DECLARE
    denied_role text;
BEGIN
    FOREACH denied_role IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
        IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = denied_role) THEN
            EXECUTE format('REVOKE ALL ON SCHEMA itinerary_private FROM %I', denied_role);
            EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA itinerary_private FROM %I', denied_role);
            EXECUTE format(
                'ALTER DEFAULT PRIVILEGES IN SCHEMA itinerary_private REVOKE ALL ON TABLES FROM %I',
                denied_role
            );
        END IF;
    END LOOP;
END;
$migration$;

COMMIT;

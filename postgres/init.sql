CREATE TABLE IF NOT EXISTS errors (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    error_message TEXT,
    execution_profile VARCHAR(20),
    created_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Europe/Athens')
);

CREATE TABLE IF NOT EXISTS skipped_computations (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL,
    reason TEXT,
    execution_profile VARCHAR(20),
    created_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Europe/Athens')
);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id VARCHAR(255) PRIMARY KEY,
    first_input_file_path VARCHAR(120) NOT NULL,
    second_input_file_path VARCHAR(120),
    evaluation_shortname VARCHAR(10) NOT NULL,
    random_seed VARCHAR(10) NOT NULL,
    error_type VARCHAR(10) NOT NULL,
    error_rate VARCHAR(3) NOT NULL,
    result NUMERIC NOT NULL,
    notes JSONB,
    execution_profile VARCHAR(20),
    created_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Europe/Athens')
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id VARCHAR(255) PRIMARY KEY,
    experiment_type VARCHAR(10) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    random_seed VARCHAR(10) NOT NULL,
    data_perfectness VARCHAR(10) NOT NULL,
    data_error VARCHAR(10) NOT NULL,
    error_rate VARCHAR(3) NOT NULL,
    generator VARCHAR(50) NOT NULL,
    training_size NUMERIC NOT NULL,
    synthetic_size NUMERIC NOT NULL,
    execution_time NUMERIC NOT NULL,
    corrupted_rows JSONB,
    corrupted_cols JSONB,
    execution_profile VARCHAR(20),
    created_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Europe/Athens')
);

CREATE INDEX idx_seed_evaluation_shortname ON evaluation_results(evaluation_shortname, random_seed)

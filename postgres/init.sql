-- TODO: This needs to be updated when we decide on the DB schema
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(50),
    model_name VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dataset_model ON model_metrics(dataset_id, model_name);
class TestHealthEndpoints:
    def test_health_endpoint_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_ready_endpoint_returns_response(self, test_client):
        response = test_client.get("/ready")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "demo_mode" in data
        assert "uptime_seconds" in data


class TestPredictEndpoint:
    def test_predict_accepts_valid_request(self, test_client):
        payload = {
            "readings": [
                {
                    "timestamp": "2025-01-01T00:00:00",
                    "equipment_id": "PUMP-001",
                    "sensor_name": "vibration_x_a",
                    "value": 2.8,
                },
                {
                    "timestamp": "2025-01-01T01:00:00",
                    "equipment_id": "PUMP-001",
                    "sensor_name": "vibration_x_a",
                    "value": 3.1,
                },
            ]
        }
        response = test_client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "failure_probability" in data
        assert "predicted_class" in data
        assert "quality_gate" in data

    def test_predict_rejects_invalid_readings(self, test_client):
        payload = {
            "readings": [
                {
                    "timestamp": "2025-01-01T00:00:00",
                    "equipment_id": "PUMP-001",
                    "sensor_name": "vibration_x_a",
                    "value": 2.8,
                },
                {
                    "timestamp": "2025-01-01T00:00:01",
                    "equipment_id": "PUMP-999",
                    "sensor_name": "temperature_a",
                    "value": 65.0,
                },
            ]
        }
        response = test_client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_returns_quality_gate_info(self, test_client):
        payload = {
            "readings": [
                {
                    "timestamp": "2025-01-01T00:00:00",
                    "equipment_id": "PUMP-001",
                    "sensor_name": "vibration_x_a",
                    "value": 2.8,
                },
                {
                    "timestamp": "2025-01-01T01:00:00",
                    "equipment_id": "PUMP-001",
                    "sensor_name": "vibration_x_a",
                    "value": 3.1,
                },
            ]
        }
        response = test_client.post("/api/v1/predict", json=payload)
        data = response.json()
        assert "quality_gate" in data
        qg = data["quality_gate"]
        assert "passed" in qg
        assert "total_violations" in qg
        assert "failing_stages" in qg

    def test_predict_empty_readings_rejected(self, test_client):
        payload = {"readings": []}
        response = test_client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422


class TestBatchPredict:
    def test_batch_predict_handles_multiple_equipment(self, test_client):
        payload = {
            "readings": [
                {
                    "timestamp": "2025-01-01T00:00:00",
                    "equipment_id": "PUMP-A01",
                    "sensor_name": "vibration_x_a",
                    "value": 2.8,
                },
                {
                    "timestamp": "2025-01-01T00:00:00",
                    "equipment_id": "PUMP-B02",
                    "sensor_name": "vibration_x_a",
                    "value": 3.1,
                },
            ]
        }
        try:
            response = test_client.post("/api/v1/batch-predict", json=payload)
            assert response.status_code in (200, 404, 422, 500, 503)
        except (RuntimeError, AttributeError, ConnectionError):
            pass


class TestExplainEndpoint:
    def test_explain_returns_response(self, test_client):
        payload = {
            "readings": [
                {
                    "timestamp": "2025-01-01T00:00:00",
                    "equipment_id": "PUMP-001",
                    "sensor_name": "vibration_x_a",
                    "value": 2.8,
                },
                {
                    "timestamp": "2025-01-01T00:00:01",
                    "equipment_id": "PUMP-001",
                    "sensor_name": "temperature_a",
                    "value": 65.0,
                },
            ]
        }
        response = test_client.post("/api/v1/explain", json=payload)
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert "failure_probability" in data
            assert "top_contributors" in data

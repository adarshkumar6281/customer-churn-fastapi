from fastapi.testclient import TestClient 
from main import app,model
client= TestClient(app)
def test_home():
    response=client.get("/")

    assert response.status_code==200
    assert "Welcome" in response.json()["Message"]

def test_health():
    response= client.get("/health")

    assert response.status_code==200
    assert response.json()["Status"]== "Healthy"
    assert response.json()["Model_loaded"] is True

def test_predict():
    input_data={"tenure":12,"monthly_charges":85.5,"total_charges":1500.00}
    response = client.post("/predict", json =input_data)

    assert response.status_code==200
    result = response.json()
    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "risk_level" in result

    assert result ["churn_prediction"] in [0,1]
    assert 0<= result["churn_probability"] <= 100
    assert result["risk_level"] in ["High","Low risk"]

def test_invalid_input():
    input_data = {"tenure":-5,"monthly_charges":85.5,"total_charges":1500.00}
    
    response = client.post("/predict",json = input_data)

    assert response.status_code==422



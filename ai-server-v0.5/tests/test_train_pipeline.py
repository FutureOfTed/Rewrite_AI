from app.pipelines.train_pipeline import execute

def test_train_pipeline_execution():
    metrics, onnx_path = execute(["data1.csv"])
    assert "rmse" in metrics
    assert "f1_score" in metrics
    assert onnx_path.endswith(".onnx")

import React, { useState } from "react";
import axios from "axios";
import "../css/DemandModel.css";

function DemandModel({ filters }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://localhost:5002/predict_demand", { 
        params: { 
          year: filters.year, 
          month: filters.month || undefined } 
      });
      setResults(res.data);
    } catch (err) {
      console.error(err);
      alert("Error predicting demand");
    }
    setLoading(false);
  };

  const handleRetrain = async () => {
    setLoading(true);
    try {
      await axios.post("http://localhost:5002/train_demand");
      alert("Demand model retrained successfully");
    } catch (err) {
      console.error(err);
      alert("Error retraining demand model");
    }
    setLoading(false);
  };

  return (
    <div className="demand-container">
      <h2 className="demand-title">Monthly Product Demand Prediction</h2>

      <div className="demand-buttons">
        <button 
          onClick={handleRetrain} 
          disabled={loading} 
          className="demand-button retrain"
        >
          {loading ? "Retraining..." : "Retrain"}
        </button>

        <button 
          onClick={handlePredict} 
          disabled={loading} 
          className="demand-button predict"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {results.length > 0 && (
        <div className="demand-table-wrapper">
          <table className="demand-table">
            <thead>
              <tr>
                <th>Year</th>
                <th>Month</th>
                <th>Quarter</th>
                <th>Predicted Demand</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, idx) => (
                <tr key={idx}>
                  <td>{r.Year}</td>
                  <td>{r.Month}</td>
                  <td>{r.Quarter}</td>
                  <td className="demand-value">{r.PredictedDemand}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default DemandModel;

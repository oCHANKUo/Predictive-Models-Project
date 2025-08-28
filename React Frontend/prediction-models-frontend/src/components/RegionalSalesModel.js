import React, { useState } from "react";
import axios from "axios";
import "../css/RegionalSalesModel.css"; 

function RegionalSalesModel({ filters }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://localhost:5001/predict_regional_sales", {
        params: { 
          months: filters.month || 6,  // use selected months, fallback = 6
          TerritoryName: filters.territory || undefined 
        }
      });
      setResults(res.data);
    } catch (err) {
      console.error(err);
      alert("Error predicting regional sales");
    }
    setLoading(false);
  };


  const handleRetrain = async () => {
    setLoading(true);
    try {
      await axios.post("http://localhost:5001/train_regional_sales");
      alert("Regional sales model retrained successfully");
    } catch (err) {
      console.error(err);
      alert("Error retraining regional sales model");
    }
    setLoading(false);
  };

  return (
    <div className="regional-container">
      <h2 className="regional-title">Regional Sales Prediction</h2>

      <div className="regional-buttons">
        <button 
          onClick={handleRetrain} 
          disabled={loading} 
          className="regional-button retrain"
        >
          {loading ? "Retraining..." : "Retrain"}
        </button>

        <button 
          onClick={handlePredict} 
          disabled={loading} 
          className="regional-button predict"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {results.length > 0 && (
        <div className="regional-table-wrapper">
          <table className="regional-table">
            <thead>
              <tr>
                <th>Territory</th>
                <th>Year</th>
                <th>Month</th>
                <th>Predicted Top Category</th>
                <th>Predicted Sales</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, idx) => (
                <tr key={idx}>
                  <td>{r.TerritoryName}</td>
                  <td>{r.Year}</td>
                  <td>{r.Month}</td>
                  <td>{r.PredictedTopCategory}</td>
                  <td className="regional-sales-value">{r.PredictedSales}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default RegionalSalesModel;

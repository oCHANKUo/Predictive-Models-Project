import React, { useState } from "react";
import axios from "axios";

function RegionalSalesModel({ filters }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://localhost:5001/predict_regional_sales", {
        params: { months: 6, TerritoryName: filters.territory || undefined }
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
    <div style={{ marginTop: "20px" }}>
      <h2>Regional Sales Prediction</h2>
      <button onClick={handleRetrain} disabled={loading}>Retrain</button>
      <button onClick={handlePredict} disabled={loading} style={{ marginLeft: "10px" }}>Predict</button>

      {results.length > 0 && (
        <table border="1" cellPadding="5" style={{ marginTop: "10px" }}>
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
                <td>{r.PredictedSales}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default RegionalSalesModel;

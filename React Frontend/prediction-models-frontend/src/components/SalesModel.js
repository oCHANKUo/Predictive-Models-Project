import React, { useState } from "react";
import axios from "axios";

function SalesModel({ filters }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
  setLoading(true);
  try {
    const res = await axios.get("http://localhost:5000/predict_sales", {
      params: { 
        months: filters.month,   // how many months into the future
        years: filters.year      // how many years into the future
      }
    });
    setResults(res.data);
  } catch (err) {
    console.error(err);
    alert("Error predicting sales");
  }
  setLoading(false);
};

  const handleRetrain = async () => {
    setLoading(true);
    try {
      await axios.post("http://localhost:5000/train_sales");
      alert("Sales model retrained successfully");
    } catch (err) {
      console.error(err);
      alert("Error retraining model");
    }
    setLoading(false);
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <h2>Monthly Sales Prediction</h2>
      <button onClick={handleRetrain} disabled={loading}>Retrain</button>
      <button onClick={handlePredict} disabled={loading} style={{ marginLeft: "10px" }}>Predict</button>

      {results.length > 0 && (
        <table border="1" cellPadding="5" style={{ marginTop: "10px" }}>
          <thead>
            <tr>
              <th>Year</th>
              <th>Month</th>
              <th>Quarter</th>
              <th>Predicted Sales</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, idx) => (
              <tr key={idx}>
                <td>{r.Year}</td>
                <td>{r.Month}</td>
                <td>{r.Quarter}</td>
                <td>{r.PredictedSales}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default SalesModel;

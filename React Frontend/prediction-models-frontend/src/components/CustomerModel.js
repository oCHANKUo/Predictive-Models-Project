import React, { useState } from "react";
import axios from "axios";

function CustomerModel({ filters }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://localhost:5003/predict_customer", {
        params: { top_n: filters.topN }
      });
      setResults(res.data);
    } catch (err) {
      console.error(err);
      alert("Error predicting customer behavior");
    }
    setLoading(false);
  };

  const handleRetrain = async () => {
    setLoading(true);
    try {
      await axios.post("http://localhost:5003/train_customer");
      alert("Customer model retrained successfully");
    } catch (err) {
      console.error(err);
      alert("Error retraining customer model");
    }
    setLoading(false);
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <h2>Customer Purchase Behavior Prediction</h2>
      <button onClick={handleRetrain} disabled={loading}>Retrain</button>
      <button onClick={handlePredict} disabled={loading} style={{ marginLeft: "10px" }}>Predict</button>

      {results.length > 0 && (
        <table border="1" cellPadding="5" style={{ marginTop: "10px" }}>
          <thead>
            <tr>
              <th>Customer Key</th>
              <th>Year</th>
              <th>Month</th>
              <th>Purchase Probability</th>
              <th>Prediction</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, idx) => (
              <tr key={idx}>
                <td>{r.CustomerKey}</td>
                <td>{r.Year}</td>
                <td>{r.Month}</td>
                <td>{r.PurchaseProbability.toFixed(2)}</td>
                <td>{r.Prediction}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default CustomerModel;

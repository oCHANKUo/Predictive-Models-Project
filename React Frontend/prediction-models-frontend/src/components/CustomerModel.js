import React, { useState } from "react";
import axios from "axios";
import "../css/CustomerModel.css";

function CustomerModel({ filters }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const params = {};
      if (filters.topN && filters.topN > 0) {
        params.top_n = filters.topN; 
        params.months = filters.month || 6;  
      }

      const res = await axios.get("http://localhost:5003/predict_customer", { params });
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
    <div className="customer-container">
      <h2 className="customer-title">Customer Purchase Behavior Prediction</h2>

      <div className="customer-buttons">
        <button 
          onClick={handleRetrain} 
          disabled={loading} 
          className="customer-button retrain"
        >
          {loading ? "Retraining..." : "Retrain"}
        </button>

        <button 
          onClick={handlePredict} 
          disabled={loading} 
          className="customer-button predict"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {results.length > 0 && (
        <div className="customer-table-wrapper">
          <table className="customer-table">
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
                  <td className="customer-value">{r.PurchaseProbability.toFixed(2)}</td>
                  <td>{r.Prediction}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default CustomerModel;

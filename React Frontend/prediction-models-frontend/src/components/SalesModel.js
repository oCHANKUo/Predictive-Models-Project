import React, { useState, useEffect  } from "react";
import axios from "axios";
import "../css/SalesModel.css";

function SalesModel({ filters }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Trigger if a year is selected
    if (filters.year) {
      handlePredict();
    }
  }, [filters.year, filters.month]);


  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://localhost:5000/predict_sales", {
      params: { 
        year: filters.year, 
        month: filters.month || null 
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
    <div className="sales-container">
      <h2 className="sales-title">Monthly Sales Prediction</h2>

      <div className="sales-buttons">
        <button 
          onClick={handleRetrain} 
          disabled={loading} 
          className="sales-button retrain"
        >
          {loading ? "Retraining..." : "Retrain"}
        </button>

        <button 
          onClick={handlePredict} 
          disabled={loading} 
          className="sales-button predict"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {results.length > 0 && (
        <div className="sales-table-wrapper">
          <table className="sales-table">
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
                  <td className="sales-sales-value">{r.PredictedSales}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default SalesModel;

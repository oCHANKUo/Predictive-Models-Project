import React, { useState } from "react";
import SalesModel from "./components/SalesModel";
import RegionalSalesModel from "./components/RegionalSalesModel";
import DemandModel from "./components/DemandModel";
import CustomerModel from "./components/CustomerModel";
import FilterPanel from "./components/FilterPanel";

function App() {
  const [filters, setFilters] = useState({
    year: 2025,
    month:"",
    territory: "",
    topN: 10
  });

  const handleFilterChange = (name, value) => {
    setFilters(prev => ({ ...prev, [name]: value }));
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Inter, sans-serif" }}>
      <h1>Predictive Models Dashboard</h1>
      <FilterPanel filters={filters} onChange={handleFilterChange} />

      <div style={{ marginTop: "30px" }}>
        <SalesModel filters={filters} />
        <RegionalSalesModel filters={filters} />
        <DemandModel filters={filters} />
        <CustomerModel filters={filters} />
      </div>
    </div>
  );
}

export default App;
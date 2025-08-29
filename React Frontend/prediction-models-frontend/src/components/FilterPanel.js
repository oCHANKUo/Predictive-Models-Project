import React, { useState } from "react";
import "../css/FilterPanel.css";

function FilterPanel({ filters, onChange }) {
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 5 }, (_, i) => currentYear - i);
  const months = Array.from({ length: 12 }, (_, i) => i + 1);

  return (
    <div className="filter-panel-container">
      <div className="filter-panel-group">
        <label>Year:</label>
        <select value={filters.year} onChange={e => onChange("year", parseInt(e.target.value))}>
          {years.map(y => <option key={y} value={y}>{y}</option>)}
        </select>
      </div>

      <div className="filter-panel-group">
        <label>Month:</label>
        <select value={filters.month} onChange={e => onChange("month", parseInt(e.target.value))}>
          <option value="">All Months</option>
          {months.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      <div className="filter-panel-group">
        <label>Territory:</label>
        <select
          value={filters.territory}
          onChange={e => onChange("territory", e.target.value)}
        >
          <option value="">All</option>
          <option value="Australia">Australia</option>
          <option value="Canada">Canada</option>
          <option value="Central">Central</option>
          <option value="France">France</option>
          <option value="Germany">Germany</option>
          <option value="Northeast">Northeast</option>
          <option value="Northwest">Northwest</option>
          <option value="Southeast">Southeast</option>
          <option value="Southwest">Southwest</option>
          <option value="United Kingdom">United Kingdom</option>
        </select>
      </div>

      <div className="filter-panel-group">
        <label>Top N Customers:</label>
        <input
          type="number"
          value={filters.topN}
          onChange={e => onChange("topN", parseInt(e.target.value))}
          min={1}
        />
      </div>
    </div>
  );
}

export default FilterPanel;

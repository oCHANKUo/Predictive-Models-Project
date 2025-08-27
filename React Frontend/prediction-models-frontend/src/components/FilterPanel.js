import React from "react";

function FilterPanel({ filters, onChange }) {
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 5 }, (_, i) => currentYear - i);
  const months = Array.from({ length: 12 }, (_, i) => i + 1);

  return (
    <div style={{ display: "flex", gap: "20px", alignItems: "center" }}>
      <div>
        <label>Year:</label>
        <select value={filters.year} onChange={e => onChange("year", parseInt(e.target.value))}>
          {years.map(y => <option key={y} value={y}>{y}</option>)}
        </select>
      </div>

      <div>
        <label>Month:</label>
        <select value={filters.month} onChange={e => onChange("month", parseInt(e.target.value))}>
          {months.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      <div>
        <label>Territory:</label>
        <input
          type="text"
          value={filters.territory}
          onChange={e => onChange("territory", e.target.value)}
          placeholder="All"
        />
      </div>

      <div>
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

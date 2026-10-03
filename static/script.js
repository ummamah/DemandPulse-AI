/* DemandPulse AI - dashboard logic */
let forecastChart = null, importanceChart = null, seasonChart = null;

const el = (id) => document.getElementById(id);

async function fetchJson(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

async function loadOptions() {
  const opts = await fetchJson("/api/options");
  el("store").innerHTML = opts.stores
    .map((s) => `<option value="${s}">${s}</option>`).join("");
  el("product").innerHTML = opts.products
    .map((p) => `<option value="${p.id}">${p.id} (${p.category})</option>`).join("");
  el("horizon").innerHTML = opts.horizons
    .map((h) => `<option value="${h}" ${h === opts.default_horizon ? "selected" : ""}>${h} Days</option>`).join("");
}

function kpiText(data) {
  const k = data.kpi;
  el("kpi-inventory").textContent = k.current_inventory;
  el("kpi-forecast").textContent = k.forecast_demand;
  el("kpi-reorder").textContent = k.recommended_reorder;
  el("kpi-horizon-label").textContent = `${k.horizon_days}-Day`;

  const riskEl = el("kpi-risk");
  riskEl.textContent = k.stockout_risk;
  riskEl.className = `kpi-value risk-${k.stockout_risk}`;
  el("kpi-risk-detail").textContent = k.risk_label.split(" ").slice(1).join(" ");

  el("inv-current-txt").textContent = `${k.current_inventory} units`;
  el("inv-demand-txt").textContent = `${k.forecast_demand} units`;
  el("inv-risk-txt").textContent = k.stockout_risk;
  const pill = el("inv-risk-pill");
  pill.textContent = k.risk_label;
  pill.className = `risk-pill risk-${k.stockout_risk}`;

  const maxVal = Math.max(k.current_inventory, k.forecast_demand, 1);
  el("inv-current-bar").style.width = `${(k.current_inventory / maxVal) * 100}%`;
  el("inv-demand-bar").style.width = `${(k.forecast_demand / maxVal) * 100}%`;

  const r = data.recommendation;
  el("rec-qty").textContent = r.recommended_reorder;
  el("rec-ltd").textContent = `${r.lead_time_demand} units`;
  el("rec-safety").textContent = `${r.safety_stock} units`;
  el("rec-gap").textContent = `${r.expected_gap} units`;
  el("rec-reason").textContent = r.reason;
}

function metrics(data) {
  const m = data.model;
  el("model-name").textContent = m.selected;
  el("as-of-date").textContent = data.as_of_date;
  const test = m.metrics[m.selected].test;
  el("met-mae").textContent = test.MAE;
  el("met-rmse").textContent = test.RMSE;
  el("met-wape").textContent = test.WAPE + "%";

  const tbody = el("compare-table").querySelector("tbody");
  tbody.innerHTML = Object.entries(m.metrics)
    .sort((a, b) => a[1].test.MAE - b[1].test.MAE)
    .map(([name, r]) =>
      `<tr class="${name === m.selected ? "selected" : ""}">
         <td>${name}</td><td>${r.test.MAE}</td><td>${r.test.RMSE}</td><td>${r.test.WAPE}%</td>
       </tr>`).join("");
}

function charts(data) {
  const labels = [...data.history.map((h) => h.date), ...data.forecast.map((f) => f.date)];
  const hist = [...data.history.map((h) => h.demand),
                ...data.forecast.map(() => null)];
  const fc = [...data.history.map(() => null),
              ...data.forecast.map((f) => f.forecast)];
  // connect the two series visually at the boundary
  const boundary = data.history.length;
  if (boundary > 0) fc[boundary - 1] = data.history[boundary - 1].demand;

  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(el("forecast-chart"), {
    type: "line",
    data: { labels,
      datasets: [
        { label: "Historical Demand", data: hist, borderColor: "#4f8cff",
          backgroundColor: "rgba(79,140,255,0.12)", fill: true, tension: 0.3,
          pointRadius: 0, borderWidth: 2 },
        { label: "Forecast", data: fc, borderColor: "#f5a623",
          backgroundColor: "rgba(245,166,35,0.10)", fill: true, tension: 0.3,
          pointRadius: 3, borderWidth: 2, borderDash: [6, 4] },
      ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8b98b3", boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#8b98b3", maxTicksLimit: 10, font: { size: 10 } }, grid: { color: "rgba(38,50,72,0.5)" } },
        y: { ticks: { color: "#8b98b3", font: { size: 10 } }, grid: { color: "rgba(38,50,72,0.5)" },
             title: { display: true, text: "Units", color: "#8b98b3", font: { size: 11 } } },
      },
    },
  });

  const imp = Object.entries(data.feature_importance).slice(0, 8);
  if (importanceChart) importanceChart.destroy();
  importanceChart = new Chart(el("importance-chart"), {
    type: "bar",
    data: { labels: imp.map(([k]) => k),
      datasets: [{ label: "Importance", data: imp.map(([, v]) => v),
        backgroundColor: "#4f8cff", borderRadius: 4 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: "#8b98b3", font: { size: 10 } }, grid: { color: "rgba(38,50,72,0.5)" } },
                y: { ticks: { color: "#8b98b3", font: { size: 10 } }, grid: { display: false } } } },
  });

  if (seasonChart) seasonChart.destroy();
  seasonChart = new Chart(el("season-chart"), {
    type: "bar",
    data: { labels: data.weekday_pattern.map((d) => d.day),
      datasets: [{ label: "Avg Demand", data: data.weekday_pattern.map((d) => d.avg),
        backgroundColor: "#2ecc8f", borderRadius: 4 }] },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: "#8b98b3", font: { size: 10 } }, grid: { display: false } },
                y: { ticks: { color: "#8b98b3", font: { size: 10 } }, grid: { color: "rgba(38,50,72,0.5)" } } } },
  });
}

async function generateForecast() {
  const store = el("store").value, product = el("product").value;
  const horizon = el("horizon").value;
  el("error").classList.add("hidden");
  try {
    const data = await fetchJson(
      `/api/forecast?store=${encodeURIComponent(store)}&product=${encodeURIComponent(product)}&horizon=${horizon}`);
    el("category").value = data.selection.category;
    kpiText(data);
    metrics(data);
    charts(data);
    el("insight-text").textContent = data.insight;
  } catch (err) {
    const box = el("error");
    box.textContent = err.message;
    box.classList.remove("hidden");
  }
}

(async function init() {
  await loadOptions();
  el("generate").addEventListener("click", generateForecast);
  await generateForecast();
})();
document
  .getElementById("backtest-form")
  .addEventListener("submit", async function (event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData);

    const response = await fetch("/run_backtest", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();
    plotResults(result);
  });

function plotResults(result) {
  const ctx = document.getElementById("portfolio-chart").getContext("2d");
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: result.all_trading_dates,
      datasets: [
        {
          label: "Portfolio Value",
          data: result.portfolio_values_over_time,
          borderColor: "blue",
          fill: false,
        },
        {
          label: "Capital",
          data: result.capital_over_time,
          borderColor: "green",
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: "time",
          time: {
            unit: "month",
          },
        },
        y: {
          beginAtZero: true,
        },
      },
    },
  });

  document.getElementById("signals").innerHTML = `
      <p>Total Portfolio Value: ${result.total_portfolio_value}</p>
      <p>CAGR: ${result.cagr}</p>
      <p>MDD: ${result.mdd}</p>
  `;
}

document
  .getElementById("backtest-form")
  .addEventListener("submit", async function (event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData);

    const considerDelistingCheckbox = document.getElementById("consider-delisting");
    if (!considerDelistingCheckbox.checked) {
      data['consider-delisting'] = 'off';
    }

    const response = await fetch("/run_backtest", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();
    displayResults(result);
  });

function displayResults(result) {
  document.getElementById("signals").innerHTML = `
      <p>Total Portfolio Value: ${result.total_portfolio_value}</p>
      <p>CAGR: ${result.cagr}</p>
      <p>MDD: ${result.mdd}</p>
      <a href="/results_of_single_test/${result.excel_file_path}">Download Trade History Excel File</a>
      <br />
      <img src="/results_of_single_test/${result.plot_file_path}" alt="Backtesting Result Graph" />
  `;
}

document.getElementById("loanForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const formData = new FormData(this);
  const data = Object.fromEntries(formData.entries());

  data.ApplicantIncome = Number(data.ApplicantIncome);
  data.CoapplicantIncome = Number(data.CoapplicantIncome);
  data.LoanAmount = Number(data.LoanAmount);
  data.Loan_Amount_Term = Number(data.Loan_Amount_Term);
  data.Credit_History = Number(data.Credit_History);

  const result = document.getElementById("result");
  result.className = "result";
  result.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });

    const output = await response.json();

    if (!response.ok) throw new Error(output.error || "Prediction failed");

    result.className = "result " + (output.approved ? "approved" : "rejected");
    result.innerHTML = `<strong>Loan ${output.prediction}</strong><br>Model confidence: ${output.probability}%`;
  } catch (error) {
    result.className = "result rejected";
    result.textContent = "Error: " + error.message;
  }
});

from flask import Flask, request, jsonify, send_file, render_template_string
import io
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fpdf import FPDF

# =========================================
# BUSINESS AI ENGINE
# =========================================
class BusinessAI:
    def __init__(self):
        self.sales = []
        self.expenses = []

    def add_data(self, sales, expenses):
        self.sales.append(sales)
        self.expenses.append(expenses)

    def remove_last(self):
        if self.sales:
            self.sales.pop()
            self.expenses.pop()
            return True
        return False

    def clear_all(self):
        self.sales.clear()
        self.expenses.clear()

    def analyze(self):
        if len(self.sales) < 2:
            return None

        months = np.arange(len(self.sales)).reshape(-1, 1)
        sales = np.array(self.sales)
        expenses = np.array(self.expenses)
        profit = sales - expenses

        # Next month prediction
        model = LinearRegression()
        model.fit(months, sales)
        next_sales_pred = model.predict([[len(self.sales)]])[0]

        last_profit = profit[-1] if len(profit) > 0 else 0
        avg_profit_increase = np.mean(np.diff(profit)) if len(profit) > 1 else 0
        next_profit_pred = last_profit + avg_profit_increase

        # Growth metrics
        growth_list = np.diff(sales) / sales[:-1] * 100
        avg_growth = np.mean(growth_list)
        profit_margin = (profit.sum() / sales.sum()) * 100

        if avg_growth > 7:
            insight = "Very strong growth"
        elif avg_growth > 2:
            insight = "Healthy growth"
        elif avg_growth > 0:
            insight = "Stable business"
        else:
            insight = "Declining business"

        suggestion = (
            "Excellent profit control"
            if profit_margin >= 30
            else "Reduce expenses or improve pricing"
        )

        highest_sales = sales.max()
        highest_sales_month = int(np.argmax(sales) + 1)

        return {
            "records": len(sales),
            "total_sales": sales.sum(),
            "total_profit": profit.sum(),
            "avg_growth": avg_growth,
            "profit_margin": profit_margin,
            "next_month_sales": next_sales_pred,
            "next_month_profit": next_profit_pred,
            "insight": insight,
            "suggestion": suggestion,
            "sales": sales.tolist(),
            "expenses": expenses.tolist(),
            "profit": profit.tolist(),
            "monthly_growth": growth_list.tolist(),
            "highest_sales": highest_sales,
            "highest_sales_month": highest_sales_month,
        }


# =========================================
# FLASK APP
# =========================================
app = Flask(__name__)
ai = BusinessAI()


# =========================================
# HTML TEMPLATE
# =========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Business Analysis AI Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --bg: #f4f6f8;
            --card-bg: #ffffff;
            --soft: #ecf0f1;
            --primary: #3498db;
            --success: #2ecc71;
            --danger: #e74c3c;
            --warning: #e67e22;
            --purple: #9b59b6;
            --teal: #1abc9c;
            --dark: #2c3e50;
        }

        * { box-sizing: border-box; }

        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #eef2f7, #f9fbfd);
            margin: 0;
            padding: 0;
            color: var(--dark);
        }

        .container {
            max-width: 1100px;
            margin: 40px auto;
            background: var(--card-bg);
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        }

        h1 {
            text-align: center;
            margin-bottom: 25px;
            letter-spacing: 0.5px;
        }

        input {
            padding: 12px;
            margin: 6px;
            border-radius: 8px;
            border: 1px solid #dcdcdc;
            width: 170px;
            outline: none;
        }

        input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(52,152,219,0.15);
        }

        button {
            padding: 11px 18px;
            margin: 6px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
        }

        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(0,0,0,0.12);
            opacity: 0.95;
        }

        button.add { background: var(--success); color: white; }
        button.remove { background: var(--warning); color: white; }
        button.clear { background: var(--danger); color: white; }
        button.analyze { background: var(--primary); color: white; }
        button.export { background: var(--purple); color: white; }
        button.pdf { background: var(--teal); color: white; }

        .cards {
            margin-top: 28px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
        }

        .card {
            padding: 16px;
            background: var(--soft);
            border-radius: 12px;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.06);
        }

        .kpi-card {
            padding: 18px;
            text-align: center;
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            border-radius: 14px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }

        .kpi-card h3 {
            margin: 6px 0;
            font-size: 0.95rem;
            font-weight: 500;
            opacity: 0.85;
        }

        .kpi-card p {
            font-size: 1.35rem;
            margin: 4px 0 0;
            font-weight: 700;
        }

        canvas {
            margin-top: 35px;
            background: white;
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 12px 25px rgba(0,0,0,0.08);
        }

        footer {
            text-align: center;
            margin-top: 26px;
            font-size: 0.9rem;
            color: #7f8c8d;
        }

        @media (max-width: 600px) {
            input { width: 100%; }
            button { width: 100%; }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Business Analysis AI Dashboard</h1>

    <div style="text-align:center;">
        <input type="number" id="sales" placeholder="Sales Amount">
        <input type="number" id="expenses" placeholder="Expense Amount"><br>

        <button class="add" onclick="action('add')">Add Data</button>
        <button class="remove" onclick="action('remove')">Remove Last</button>
        <button class="clear" onclick="action('clear')">Clear All</button>
        <button class="analyze" onclick="action('analyze')">Analyze Business</button>
        <button class="export" onclick="window.location='/export_csv'">Export CSV</button>
        <button class="pdf" onclick="window.location='/export_pdf'">Download PDF Report</button>
    </div>

    <div class="cards" id="kpi-cards"></div>
    <div class="cards" id="cards"></div>

    <canvas id="trendChart" width="950" height="400"></canvas>
    <canvas id="growthChart" width="950" height="200"></canvas>

    <footer>© Affan's Business AI</footer>
</div>

<script>
    let trendChart;
    let growthChart;

    async function action(act) {
        const sales = parseFloat(document.getElementById('sales').value) || 0;
        const expenses = parseFloat(document.getElementById('expenses').value) || 0;

        const res = await fetch('/api', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: act, sales: sales, expenses: expenses })
        });

        const data = await res.json();

        if (!data) {
            document.getElementById('cards').innerHTML = '<div class="card">Add at least 2 records</div>';
            updateTrendChart([], [], [], 0, 0);
            updateGrowthChart([]);
            return;
        }

        document.getElementById('kpi-cards').innerHTML = `
            <div class="kpi-card"><h3>Total Sales</h3><p>$${data.total_sales.toFixed(2)}</p></div>
            <div class="kpi-card"><h3>Total Profit</h3><p>$${data.total_profit.toFixed(2)}</p></div>
            <div class="kpi-card"><h3>Avg Growth</h3><p>${data.avg_growth.toFixed(2)}%</p></div>
            <div class="kpi-card"><h3>Profit Margin</h3><p>${data.profit_margin.toFixed(2)}%</p></div>
            <div class="kpi-card"><h3>Highest Sales Month</h3><p>Month ${data.highest_sales_month} ($${data.highest_sales.toFixed(2)})</p></div>
        `;

        document.getElementById('cards').innerHTML = `
            <div class="card"><strong>Records:</strong> ${data.records}</div>
            <div class="card"><strong>Next Month Sales Prediction:</strong> $${data.next_month_sales.toFixed(2)}</div>
            <div class="card"><strong>Next Month Profit Prediction:</strong> $${data.next_month_profit.toFixed(2)}</div>
            <div class="card"><strong>Insight:</strong> ${data.insight}</div>
            <div class="card"><strong>Suggestion:</strong> ${data.suggestion}</div>
        `;

        updateTrendChart(data.sales, data.expenses, data.profit, data.next_month_sales, data.next_month_profit);
        updateGrowthChart(data.monthly_growth);
    }

    function updateTrendChart(sales, expenses, profit, nextSales, nextProfit) {
        const ctx = document.getElementById('trendChart').getContext('2d');
        if (trendChart) trendChart.destroy();

        const labels = sales.map((_, i) => `Month ${i + 1}`).concat([`Month ${sales.length + 1}`]);
        const extendedSales = sales.concat([nextSales]);
        const extendedExpenses = expenses.concat([null]);
        const extendedProfit = profit.concat([nextProfit]);

        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Sales', data: extendedSales, borderColor: '#2ecc71', backgroundColor: 'rgba(46,204,113,0.2)', tension: 0.3 },
                    { label: 'Expenses', data: extendedExpenses, borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.2)', tension: 0.3 },
                    { label: 'Profit', data: extendedProfit, borderColor: '#3498db', backgroundColor: 'rgba(52,152,219,0.2)', tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'top' } },
                scales: { y: { beginAtZero: true } },
                interaction: { mode: 'index', intersect: false }
            }
        });
    }

    function updateGrowthChart(monthly_growth) {
        const ctx = document.getElementById('growthChart').getContext('2d');
        if (growthChart) growthChart.destroy();

        growthChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: monthly_growth.map((_, i) => `Month ${i + 2}`),
                datasets: [{ label: 'Monthly Growth %', data: monthly_growth, backgroundColor: '#f1c40f' }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } },
                plugins: { legend: { display: false } }
            }
        });
    }
</script>
</body>
</html>
"""


# =========================================
# FLASK ROUTES
# =========================================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api', methods=['POST'])
def api():
    data = request.json
    action = data.get('action')
    sales = float(data.get('sales', 0))
    expenses = float(data.get('expenses', 0))

    if action == 'add':
        ai.add_data(sales, expenses)
    elif action == 'remove':
        ai.remove_last()
    elif action == 'clear':
        ai.clear_all()

    analysis = ai.analyze()
    if not analysis:
        return jsonify(None)
    return jsonify(analysis)


@app.route('/export_csv')
def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Sales', 'Expenses', 'Profit'])

    for s, e in zip(ai.sales, ai.expenses):
        writer.writerow([s, e, s - e])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        download_name='business_data.csv',
        as_attachment=True,
    )


@app.route('/export_pdf')
def export_pdf():
    analysis = ai.analyze()
    if not analysis:
        return 'Add at least 2 records to export PDF'

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Business Analysis AI Report', 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Total Sales: ${analysis['total_sales']:.2f}", 0, 1)
    pdf.cell(0, 8, f"Total Profit: ${analysis['total_profit']:.2f}", 0, 1)
    pdf.cell(0, 8, f"Average Growth: {analysis['avg_growth']:.2f}%", 0, 1)
    pdf.cell(0, 8, f"Profit Margin: {analysis['profit_margin']:.2f}%", 0, 1)
    pdf.cell(0, 8, f"Next Month Sales Prediction: ${analysis['next_month_sales']:.2f}", 0, 1)
    pdf.cell(0, 8, f"Next Month Profit Prediction: ${analysis['next_month_profit']:.2f}", 0, 1)
    pdf.cell(0, 8, f"Highest Sales Month: Month {analysis['highest_sales_month']} (${analysis['highest_sales']:.2f})", 0, 1)
    pdf.cell(0, 8, f"Insight: {analysis['insight']}", 0, 1)
    pdf.cell(0, 8, f"Suggestion: {analysis['suggestion']}", 0, 1)

    plt.figure(figsize=(8, 4))
    months = list(range(1, len(ai.sales) + 1))
    plt.plot(months, ai.sales, label='Sales', marker='o')
    plt.plot(months, ai.expenses, label='Expenses', marker='o')
    plt.plot(months, np.array(ai.sales) - np.array(ai.expenses), label='Profit', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.title('Sales, Expenses, Profit Trend')
    plt.legend()
    plt.tight_layout()

    chart_file = 'trend_chart.png'
    plt.savefig(chart_file)
    plt.close()

    pdf.image(chart_file, x=10, w=190)

    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    if os.path.exists(chart_file):
        os.remove(chart_file)

    return send_file(pdf_buffer, download_name='business_report.pdf', as_attachment=True)


# =========================================
# RUN APP
# =========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

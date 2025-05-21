import http.client
import json
import time
import pandas as pd

# Function to check the task status
def check_task_status(task_id):
    conn = http.client.HTTPSConnection("crunchbase-scraper.p.rapidapi.com")
    
    # Set up the URL with the task ID to check status
    url = f"/api/b2b/crunchbase-scraper/monitor-status/{task_id}"
    
    headers = {
        'x-rapidapi-key': "80b1845ac5msh932ee5d309d9107p13f56ajsnfda1b9343e0e",  # Your API key
        'x-rapidapi-host': "crunchbase-scraper.p.rapidapi.com"
    }
    
    conn.request("GET", url, headers=headers)
    res = conn.getresponse()
    data = res.read()
    
    if res.status == 200:
        return json.loads(data.decode("utf-8"))
    else:
        print(f"Failed to get status with status code: {res.status}")
        return None

# Function to fetch the initial data
def fetch_data():
    conn = http.client.HTTPSConnection("crunchbase-scraper.p.rapidapi.com")
    payload = '[{"url":"https://www.crunchbase.com/organization/aisci"}]'

    headers = {
        'x-rapidapi-key': "80b1845ac5msh932ee5d309d9107p13f56ajsnfda1b9343e0e",
        'x-rapidapi-host': "crunchbase-scraper.p.rapidapi.com",
        'Content-Type': "application/json"
    }

    conn.request("POST", "/api/b2b/crunchbase-scraper/companies", payload, headers)
    
    res = conn.getresponse()
    data = res.read()
    
    # Handle different status codes from API
    if res.status == 200:
        response_data = json.loads(data.decode("utf-8"))
        return response_data
    elif res.status == 202:
        # If status code is 202, it means the task is still processing
        response = json.loads(data.decode("utf-8"))
        task_id = response.get("task_id")
        print(f"Task is processing. Task ID: {task_id}")
        # Monitor the task status and check until it's ready
        while True:
            task_status = check_task_status(task_id)
            if task_status and task_status.get("status") == "completed":
                print("Task completed. Data is ready.")
                return task_status.get("data")
            elif task_status and task_status.get("status") == "failed":
                print("Task failed.")
                return None
            else:
                print("Waiting for task to complete...")
                time.sleep(60)  # Wait for 30 seconds before checking again
    else:
        print(f"Request failed with status code: {res.status}")
        return None

# Function to store data into an Excel file
def store_data_to_excel(data):
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    
    # Save the DataFrame to an Excel file
    df.to_excel('company_data.xlsx', index=False)

    print("Data has been saved to company_data.xlsx")

# Function to fetch data with retry for rate limiting
def fetch_data_with_retry():
    retry_count = 0
    while retry_count < 10:  # Retry up to 10 times for robustness
        data = fetch_data()
        if data:
            return data
        else:
            retry_count += 1
            # Check if rate limit error is encountered
            print(f"Rate limit exceeded. Retrying in {60 ** retry_count} seconds...")
            time.sleep(60 ** retry_count)  # Exponential backoff
    print("Failed to fetch data after retries.")
    return None

# Main function to handle data extraction and saving
def process_data():
    data = fetch_data_with_retry()

    if data:
        print("Fetched data:", json.dumps(data, indent=4))  # Print the raw data to inspect it
        
        company_info = []
        
        for company in data:
            industries = ', '.join([industry['value'] for industry in company.get('industries', [])])
            location = ', '.join([loc['name'] for loc in company.get('location', [])])
            
            company_info.append({
                'Company Name': company.get('name', 'N/A'),
                'Industries': industries if industries else 'N/A',
                'Location': location if location else 'N/A',
                'Revenue': 'N/A' # No revenue field in the response
            })

        # Store data to Excel
        store_data_to_excel(company_info)
    else:
        print("Failed to fetch data.")
import json
from fpdf import FPDF
import pandas as pd

# Function to generate PDF report
def generate_pdf_report(data, model_metrics, file_name="classification_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title of the Report
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Company Classification Report", ln=True, align='C')
    pdf.ln(10)

    # Add model performance metrics
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Model Performance Metrics", ln=True)
    pdf.cell(200, 10, txt=f"Accuracy: {model_metrics['accuracy']}", ln=True)
    pdf.cell(200, 10, txt=f"Precision: {model_metrics['precision']}", ln=True)
    pdf.cell(200, 10, txt=f"Recall: {model_metrics['recall']}", ln=True)
    pdf.cell(200, 10, txt=f"F1 Score: {model_metrics['f1_score']}", ln=True)
    pdf.ln(10)

    # Add classification results
    pdf.cell(200, 10, txt="Classification Results", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, 'Company Name', 1)
    pdf.cell(40, 10, 'Industry', 1)
    pdf.cell(40, 10, 'Prediction', 1)
    pdf.ln()

    pdf.set_font("Arial", size=12)
    for company in data:
        pdf.cell(40, 10, company['name'], 1)
        pdf.cell(40, 10, company['industry'], 1)
        pdf.cell(40, 10, company['prediction'], 1)
        pdf.ln()

    # Save the PDF
    pdf.output(file_name)
    print(f"PDF Report saved as {file_name}")

# Example data: Replace this with actual results from your prediction
company_data = [
    {"name": "Company A", "industry": "Tech", "prediction": "Raised Money"},
    {"name": "Company B", "industry": "Retail", "prediction": "Other"},
    {"name": "Company C", "industry": "Healthcare", "prediction": "Raised Money"}
]

# Example model performance metrics: Replace this with actual model evaluation metrics
model_metrics = {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.78,
    "f1_score": 0.80
}

# Generate and save the PDF report
generate_pdf_report(company_data, model_metrics)

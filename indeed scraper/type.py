import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting the Enhanced Job Search Comparison Tool in Jupyter Notebook...")

def scrape_jobs():
    # Simulated job data; replace this with real scraped data if available
    jobs = [
        {"platform": "Indeed", "title": "Software Engineer", "company": "TechCorp", "location": "New York", "salary": 120000, "type": "Full-time"},
        {"platform": "LinkedIn", "title": "Data Scientist", "company": "Data Inc.", "location": "San Francisco", "salary": 150000, "type": "Full-time"},
        {"platform": "Indeed", "title": "Backend Developer", "company": "CodeBase", "location": "Austin", "salary": 110000, "type": "Part-time"},
        {"platform": "LinkedIn", "title": "Frontend Developer", "company": "DesignLab", "location": "Remote", "salary": 105000, "type": "Contract"},
        {"platform": "Glassdoor", "title": "Product Manager", "company": "BizSolutions", "location": "Seattle", "salary": 130000, "type": "Full-time"},
        {"platform": "Indeed", "title": "Software Engineer", "company": "TechSoft", "location": "Remote", "salary": 115000, "type": "Full-time"},
    ]
    logging.info("Scraped job listings successfully.")
    return pd.DataFrame(jobs)

jobs_df = scrape_jobs()
jobs_df
def normalize_data(jobs_df):
    jobs_df["salary"] = jobs_df["salary"].astype(float)
    jobs_df["location"] = jobs_df["location"].str.lower()
    jobs_df["type"] = jobs_df["type"].str.lower()
    logging.info("Normalized job data.")
    return jobs_df

jobs_df = normalize_data(jobs_df)
jobs_df
def compare_jobs(jobs_df, criteria, preferred_location=None, salary_threshold=None):
    if criteria == "highest_salary":
        return jobs_df.sort_values(by="salary", ascending=False)
    elif criteria == "location" and preferred_location:
        return jobs_df[jobs_df["location"].str.contains(preferred_location.lower())]
    elif criteria == "platform":
        return jobs_df.groupby("platform").size().reset_index(name="job_count")
    elif criteria == "salary_above" and salary_threshold:
        return jobs_df[jobs_df["salary"] > salary_threshold]
    else:
        logging.warning("Invalid criteria. Returning all jobs.")
        return jobs_df

# Example: Compare by highest salary
compare_jobs(jobs_df, criteria="highest_salary")
def send_alert(jobs_df, criteria):
    if criteria == "salary_above":
        print("\nAlert: Jobs with a high salary:")
        print(jobs_df)

# Example: Alert for salary above 120000
filtered_jobs = compare_jobs(jobs_df, criteria="salary_above", salary_threshold=120000)
send_alert(filtered_jobs, criteria="salary_above")
def visualize_data(jobs_df, criteria):
    if criteria == "platform":
        platform_counts = jobs_df.groupby("platform").size()
        platform_counts.plot(kind="bar", color="skyblue", title="Jobs by Platform")
        plt.xlabel("Platform")
        plt.ylabel("Job Count")
        plt.xticks(rotation=45)
        plt.show()
    elif criteria == "highest_salary":
        jobs_df.plot(kind="bar", x="title", y="salary", color="green", title="Jobs by Salary")
        plt.xlabel("Job Title")
        plt.ylabel("Salary")
        plt.xticks(rotation=45)
        plt.show()

# Example: Visualize by platform
visualize_data(jobs_df, criteria="platform")
def save_results(jobs_df, file_name="job_comparison.xlsx"):
    jobs_df.to_excel(file_name, index=False)
    logging.info(f"Comparison results saved to {file_name}.")


# Example: Save results
save_results(jobs_df)
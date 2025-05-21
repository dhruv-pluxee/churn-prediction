import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlparse
from together import Together  # Import the Together AI library
# Import GoogleNews, as it's used in fetch_news
from pygooglenews import GoogleNews
import io  # For handling file-like objects


# --- Configuration ---
# Replace with your API key
together_client = Together(
    api_key="a510758b9bff7bf393548b99848a45972486dd1d699eb86a5e7735d2339c1d8c")

# --- Prompts ---
# Define the prompt for analyzing individual articles (short summary with risk and reason)
PROMPT_INDIVIDUAL_ANALYSIS = """Analyze the following text for information that could indicate potential reasons for client churn for an employee benefits company in India.

**Text:**
{provided_text}

Provide a concise analysis (at most 2 lines). In the first line, state the level of risk for churn (e.g., "High Risk," "Medium Risk," "Low Risk," "No Churn Risk Indicated"). In the second line, briefly explain the major reason(s) for the churn risk based on the categories below. If no relevant information is found, state "No Churn Risk Indicated."

**Categories for Reasons:**
I. Corporate Restructuring (Mergers, Acquisitions, Joint Ventures, IPOs, Entity Realignment, Rebranding, Consolidation, Subsidiary changes)
II. Business Discontinuity (Closures, Market Exits, Bankruptcy, Operational Suspensions, Business Model Pivots)
III. Strategic Policy Changes (Benefits Strategy Transformation, Leadership Changes impacting strategy, Cost Optimization related to benefits)
IV. Financial Constraints (Cash Flow Issues, Cost-Cutting impacting benefits, Budget Reallocation away from benefits)
V. Employment Structure Changes (Workforce Reorganization, Shifts to contractual work, Remote work transitions impacting benefits)
VI. Regulatory & Compliance Factors (India Specific: Changes in tax policy, GST, labor codes, social security impacting benefits)
VII. Competitive Market Dynamics (Switched vendor, New platform, Competitor activity, Pricing, Market share, Disruption, Value proposition)
VIII. Technological Transitions (Digital transformation, HRMS integration, API, Analytics, Mobile app, Platform upgrade)
IX. Service Delivery Issues (Onboarding delay, Tech issues, Merchant issue, Support problem, Delivery delay, Reimbursement issue)
X. Employee Engagement (Low adoption, User experience, Employee feedback, Generation gap, Hybrid work, Usage drop)
"""

# Define the prompt for summarizing combined analyses
PROMPT_COMBINED_ANALYSIS = """Given the individual analyses of news articles related to a company and potential client churn, provide an overall summary (at most 4 lines).

**Individual Article Analyses:**
{individual_analyses_summary}

In the first line, state the overall risk level for churn for the company (e.g., "Overall High Risk," "Overall Medium Risk," "Overall Low Risk," "Overall No Churn Risk Indicated"). In the subsequent lines, summarize the major reasons for this overall risk, drawing from the categories mentioned in the individual analyses. Be concise and focus on the most impactful reasons across all articles. If no relevant information is found across all articles, state "Overall No Churn Risk Indicated."
"""

# --- Functions (mostly the same as your original code, but adapted for Streamlit) ---
# Cell 5: Define analyze_text function


def analyze_text(company_name, provided_text, prompt_template, together_client):
    """Analyzes the provided text for churn indicators using Together AI."""
    prompt = prompt_template.format(
        company_name=company_name, provided_text=provided_text)
    try:
        print(f"Cell 5: Analyzing text for company: {company_name}")
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {
                  "role": "user",
                    "content": prompt
                }
            ])
        output = response.choices[0].message.content  # get response

        # print(f"Cell 5: Response from Together AI: {output}")
        if output:
            return output
        else:
            return f"Unexpected response: {output}"
    except Exception as e:
        print(f"Error querying Together AI for {company_name}: {e}")
        return None

# Cell 6: Define fetch_news function


def fetch_news(company_name, from_date, to_date, max_articles=10, queries=None, allowed_domains=None):
    """
    Fetches news articles for a given company using the pygooglenews library.

    Args:
        company_name (str): The name of the company to search for.
        from_date (datetime): The start date for the search.
        to_date (datetime): The end date for the search.
        max_articles (int): The maximum number of articles to fetch per query.
        queries (list, optional): A list of search queries. If None, defaults to company_name.
        allowed_domains (list, optional): A list of allowed domains.

    Returns:
        list: A list of news articles, or None if an error occurs. Each article is a dictionary.
    """
    gn = GoogleNews(lang='en', country='IN')
    results = []
    if queries is None:
        queries = [company_name]
    print(
        f"Cell 6: Fetching news for {company_name}, from: {from_date}, to: {to_date}, max_articles: {max_articles}, queries: {queries}, allowed_domains: {allowed_domains}")
    try:
        for i in range(0, len(queries), 3):
            group_queries = queries[i:i+3]
            combined_query = " OR ".join(group_queries)
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            search_results = gn.search(
                combined_query, from_=from_date_str, to_=to_date_str)
            # print(f"Cell 6: Search results for query '{combined_query}': {search_results}")
            if search_results and 'entries' in search_results:
                if allowed_domains:
                    filtered_results = []
                    for article in search_results['entries']:
                        source_link = article.get('source', {}).get('href', '')
                        parsed_uri = urlparse(source_link)
                        # Remove www.
                        domain = parsed_uri.netloc.replace('www.', '')
                        # Check if any allowed domain is in the source domain
                        if any(d in domain for d in allowed_domains):
                            filtered_results.append(article)

                    num_found = len(search_results['entries'])
                    num_selected = len(filtered_results)
                    print(
                        f"Cell 6: {num_found} articles found, {num_selected} articles selected after domain filtering.")
                    if num_selected == 0:
                        # If no articles from allowed domains, add the top article as a fallback
                        if search_results['entries']:
                            print(
                                "Cell 6: No allowed domain articles found, adding top article as fallback.")
                            results.extend(search_results['entries'][:1])
                    elif num_selected > 0:
                        # Take top 3 from filtered
                        results.extend(filtered_results[:3])
                else:
                    num_found = len(search_results['entries'])
                    num_selected = min(num_found, max_articles)
                    print(
                        f"Cell 6: {num_found} articles found, {num_selected} articles selected.")
                    results.extend(search_results['entries'][:max_articles])
            else:
                print(
                    f"Cell 6: No results or 'entries' not found for query '{combined_query}'")
        print(f"Cell 6: Returning {len(results)} articles")
        return results[:10]  # Ensure total articles returned is at most 10
    except Exception as e:
        print(
            f"Cell 6: Error fetching news for {company_name} using pygooglenews: {e}")
        return None

# Cell 7: Define process_article function


def process_article(article):
    """Processes a single news article from pygooglenews's response."""
    # print("Cell 7: Processing news article")
    if article.get('summary'):
        # print("Cell 7: Article has summary")
        return article['summary']
    elif article.get('title'):
        # print("Cell 7: Article has title")
        return article['title']
    else:
        # print("Cell 7: Article has no summary or title")
        return ""

# Cell 8: Define analyze_news function


def analyze_news(company_name, from_date, to_date, max_articles=10, queries=None, allowed_domains=None):
    """
    Fetches news articles for a company and analyzes them for churn indicators.

    Args:
        company_name (str): Name of the company.
        from_date (datetime): Start date for news search.
        to_date (datetime): End date for news search.
        max_articles (int): The maximum number of articles to fetch.
        queries (list, optional): A list of search queries.
        allowed_domains (list, optional): A list of allowed domains

    Returns:
        dict: A dictionary containing the analysis results for the company,
              or None if no analysis was possible.
    """
    print(f"Cell 8: Analyzing news for churn for company: {company_name}")
    if queries is None:
        queries = [company_name]
    all_articles = fetch_news(company_name, from_date,
                              to_date, max_articles, queries, allowed_domains)
    if all_articles is None:
        print(
            f"Cell 8: Failed to fetch news for {company_name} using pygooglenews. Skipping.")
        return None

    print(
        f"Cell 8: Fetched {len(all_articles)} articles for {company_name} from pygooglenews.")

    if not all_articles:
        print("Cell 8: No articles found.")
        return {"overall_summary": "No relevant news articles found."}

    individual_analyses_list = []
    # This will be the text passed to the combined analysis prompt
    combined_analysis_text_for_model = ""

    for i, article in enumerate(all_articles):
        article_text = process_article(article)
        article_url = article.get('link', 'No URL available')
        # print(f"Cell 8: Analyzing article {i+1} from URL: {article_url}")

        if article_text:
            # Use PROMPT_INDIVIDUAL_ANALYSIS for each article
            analysis_result = analyze_text(
                company_name, article_text, PROMPT_INDIVIDUAL_ANALYSIS, together_client)
            individual_analyses_list.append({
                "url": article_url,
                "analysis": analysis_result
            })
            combined_analysis_text_for_model += f"Article {i+1} Analysis:\n{analysis_result}\n\n"
        else:
            print("Cell 8: Article has no text.")
            no_text_analysis = "No Churn Risk Indicated (No text in article)."
            individual_analyses_list.append({
                "url": article_url,
                "analysis": no_text_analysis
            })
            combined_analysis_text_for_model += f"Article {i+1} Analysis:\n{no_text_analysis}\n\n"

    # Now, generate the combined summary using the new prompt
    if individual_analyses_list:
        print("Cell 8: Generating combined analysis.")
        combined_prompt = PROMPT_COMBINED_ANALYSIS.format(
            individual_analyses_summary=combined_analysis_text_for_model.strip())
        # Pass combined_prompt as provided_text
        overall_summary_result = analyze_text(
            company_name, combined_prompt, "{provided_text}", together_client)
    else:
        overall_summary_result = "Overall No Churn Risk Indicated."
    print("Cell 8: Combined analysis complete.")
    return {"individual_analyses": individual_analyses_list, "overall_summary": overall_summary_result}


# Cell 9: Define main function
def main(company_names):
    """Main function to orchestrate the news fetching and analysis."""
    results = {}
    today = datetime.today()
    one_month_ago = today - timedelta(days=100)
    max_articles = 10

    churn_keywords = {
        "Corporate Restructuring": [
            "merger", "acquisition", "investment", "joint venture", "IPO", "restructuring",
            "realignment", "rebranding", "subsidiary", "consolidation"
        ],
        "Business Discontinuity": [
            "shutdown", "closed", "bankruptcy", "insolvency", "pivot", "market exit"
        ],
        "Strategic Policy Changes": [
            "benefits withdrawn", "benefits discontinued", "centralization",
            "new CEO", "cost cutting", "budget cuts", "strategy shift"
        ],
        "Financial Constraints": [
            "payroll issue", "financial loss", "cost pressure", "cash flow", "budget reallocation"
        ],
        "Employment Structure Changes": [
            "employee transfer", "contractual workforce", "remote work",
            "layoffs", "furloughs", "downsizing"
        ],
        "Regulatory & Compliance": [
            "tax policy", "labor law", "income tax", "GST change", "budget amendment", "social security"
        ],
        "Competitive Market Dynamics": [
            "switched vendor", "new platform", "competitor", "pricing", "market share",
            "disruption", "value proposition"
        ],
        "Technological Transitions": [
            "digital transformation", "HRMS integration", "API", "analytics",
            "mobile app", "platform upgrade"
        ],
        "Service Delivery Issues": [
            "onboarding delay", "tech issues", "merchant issue", "support problem",
            "delivery delay", "reimbursement issue"
        ],
        "Employee Engagement": [
            "low adoption", "user experience", "employee feedback",
            "generation gap", "hybrid work", "usage drop"
        ]
    }

    allowed_domains = [
        "livemint.com", "economictimes.indiatimes.com", "business-standard.com",
        "thehindubusinessline.com", "financialexpress.com", "ndtvprofit.com",
        "zeebiz.com", "moneycontrol.com", "bloombergquint.com",
        "cnbctv18.com", "businesstoday.in", "indianexpress.com",
        "thehindu.com", "reuters.com", "businesstraveller.com",
        "sify.com", "telegraphindia.com", "outlookindia.com",
        "firstpost.com", "pulse.zerodha.com", "ndtvprofit.com",
        "ddnews.gov.in", "newsonair.gov.in", "pib.gov.in",
        "niti.gov.in", "rbi.org.in", "sebi.gov.in",
        "dpiit.gov.in", "investindia.gov.in", "indiabriefing.com",
        "Taxscan.in", "bwbusinessworld.com", "inc42.com",
        "yourstory.com", "vccircle.com", "entrackr.com",
        "the-ken.com", "linkedin.com", "mca.gov.in"
    ]

    processed_allowed_domains = [domain.replace(
        "www.", "") for domain in allowed_domains]

    st.sidebar.write(
        f"Analysis Period: {one_month_ago.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}")
    st.sidebar.write(f"Max Articles per Query: {max_articles}")

    for company in company_names:
        st.write(f"\n--- Starting Analysis for {company} ---")
        queries = [company] + [f"{company} {keyword}" for category_keywords in churn_keywords.values()
                               for keyword in category_keywords]
        company_analysis = analyze_news(
            company, one_month_ago, today, max_articles, queries, processed_allowed_domains)
        results[company] = company_analysis if company_analysis else {
            "overall_summary": "Analysis failed."}
        st.write(f"--- Analysis Complete for {company} ---")

    return results


# --- Streamlit App Layout ---
st.title("Company Churn Risk Analysis")

# File uploader widget
uploaded_file = st.file_uploader(
    "Upload your 'company_names.csv' file", type=["csv"])

company_names = []
if uploaded_file is not None:
    try:
        company_df = pd.read_csv(uploaded_file)
        if "CompanyName" in company_df.columns:
            company_names = company_df["CompanyName"].tolist()
            st.success(
                f"Loaded {len(company_names)} companies from '{uploaded_file.name}'.")
        else:
            st.error("Error: The uploaded CSV must contain a 'CompanyName' column.")
            company_names = []  # Reset company_names if column is missing
    except Exception as e:
        st.error(
            f"Error reading CSV file: {e}. Please ensure it's a valid CSV.")
        company_names = []  # Reset company_names on error
else:
    st.info("Please upload a CSV file with a 'CompanyName' column to proceed.")


if st.button("Start Analysis"):
    if not company_names:
        st.warning("No company names loaded. Please upload a valid CSV file.")
    else:
        with st.spinner("Running analysis... This might take a few moments for each company."):
            analysis_results = main(company_names)

        st.success("Analysis Complete!")

        # Display the results
        for company, analysis in analysis_results.items():
            st.markdown(f"## :office: {company}")
            st.markdown("### Overall Churn Risk Summary")
            st.info(analysis.get("overall_summary",
                    "No overall analysis available."))

            st.markdown("### Individual Article Analyses")
            if analysis.get("individual_analyses"):
                for i, article_analysis in enumerate(analysis["individual_analyses"]):
                    st.markdown(f"#### Article {i+1}")
                    st.markdown(f"**URL:** [Link]({article_analysis['url']})")
                    st.write(f"**Analysis:** {article_analysis['analysis']}")
                    st.markdown("---")
            else:
                st.write("No individual articles found for analysis.")

        # Export to Excel
        data_for_df = []
        for company, analysis in analysis_results.items():
            company_data = {"Company": company, "Overall Summary": analysis.get(
                "overall_summary", "No analysis available")}
            for i, article_analysis in enumerate(analysis.get("individual_analyses", [])):
                company_data[f"Article {i+1} URL"] = article_analysis["url"]
                company_data[f"Article {i+1} Analysis"] = article_analysis["analysis"]
            data_for_df.append(company_data)

        if data_for_df:
            df_results = pd.DataFrame(data_for_df)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_file_name = f"churn_analysis_results_{timestamp}.xlsx"

            # Streamlit's way to allow downloading a file
            excel_buffer = io.BytesIO()
            df_results.to_excel(excel_buffer, index=False, engine='xlsxwriter')
            excel_buffer.seek(0)  # Rewind the buffer to the beginning

            st.download_button(
                label="Download Results as Excel",
                data=excel_buffer,
                file_name=excel_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("Results are ready for download.")
        else:
            st.warning("No data to export to Excel.")

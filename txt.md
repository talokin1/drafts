1. HNWI External Clients

Project Description
Developed a machine learning solution for identifying and prioritizing potential High-Net-Worth Individuals among external non-client prospects. The solution used external behavioral and asset-related data to estimate financial potential and generate ranked acquisition lists for affluent and private banking segments.

Customer
CEE Banking & Financial Services company

Involvement Duration
5 months

Project Role
Data Scientist

Responsibilities

Designed an end-to-end ML pipeline for identifying HNWI and affluent prospects among external individuals
Collected and processed external AutoRIA data and engineered financial-potential proxy features based on vehicle value, ownership patterns, vehicle class, and premium brands
Developed a multi-class classification approach to separate HNWI, Premium, and Mass-market prospects
Addressed strong class imbalance and optimized decision thresholds and Top-K ranking for acquisition use cases
Processed more than 260K external candidates and generated a prioritized shortlist of high-potential prospects for business validation
Worked with business stakeholders to define validation criteria and translate model scores into actionable acquisition lists

Project Team Size
4 team members (Data Scientist + business stakeholders)

Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, SQL, Oracle, Jupyter, Excel

2. HNWI Golden Model

Project Description
Developed a propensity model for identifying retail banking clients with a high probability of becoming Golden / affluent-segment customers. The model analyzed clients’ financial activity and behavioral patterns to support targeted acquisition and segment migration campaigns.

Customer
CEE Banking & Financial Services company

Involvement Duration
4 months

Project Role
Data Scientist

Responsibilities

Designed a supervised machine learning model for predicting clients with high potential to transition into the Golden banking segment
Built behavioral and financial features from transaction history, balances, product usage, income indicators, and client activity
Performed feature selection, class-imbalance handling, cross-validation, and probability calibration
Optimized the decision threshold based on business requirements and campaign capacity
Generated a prioritized acquisition list of more than 4K clients for business teams
Analyzed prediction distributions and existing Golden clients to validate the model and explain key behavioral patterns to stakeholders

Project Team Size
3 team members (Data Scientist + retail business stakeholders)

Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, SQL, Oracle, Jupyter, Excel, Power BI

3. Acquiring Potential Model

Project Description
Developed a machine learning model for identifying corporate clients and prospects with high potential for merchant acquiring products. The solution prioritized companies based on their expected propensity and commercial value for acquiring services.

Customer
CEE Banking & Financial Services company

Involvement Duration
3 months

Project Role
Data Scientist

Responsibilities

Analyzed corporate transaction and company-level data to identify behavioral patterns associated with acquiring-product adoption
Designed features based on company turnover, industry, transaction structure, business size, cash flows, and existing banking products
Built and compared classification models for acquiring propensity scoring
Performed model validation, threshold optimization, and feature-importance analysis
Developed a ranking methodology combining propensity scores with estimated business potential
Delivered prioritized corporate client lists for targeted sales campaigns and presented model logic and limitations to business stakeholders

Project Team Size
4 team members (Data Scientist + corporate banking stakeholders)

Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, SQL, Oracle, Jupyter, Power BI, Excel

4. ZKP x CC Model

Project Description
Developed a data-driven acquisition solution combining salary-project targeting with consumer lending opportunities. The system identified companies with high salary-project potential and estimated additional cash-loan and refinancing opportunities among their employees.

Customer
CEE Banking & Financial Services company

Involvement Duration
4 months

Project Role
Data Scientist

Responsibilities

Designed an ML approach for prioritizing external companies for salary-project acquisition
Trained models on existing banking relationships and transferred learned patterns to external corporate prospects
Combined internal banking data with external company information to create company-level behavioral and financial features
Estimated salary-project propensity together with the potential number and volume of employee lending opportunities
Developed ranking logic to identify companies with the highest combined acquisition and cross-selling potential
Prepared prioritized company lists for salary-project and consumer-lending teams and worked with business stakeholders on model validation

Project Team Size
4 team members (Data Scientist + salary-project and lending stakeholders)

Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, SQL, Oracle, Jupyter, Excel

5. YouControl x uBKI Data Parser

Project Description
Developed a scalable data collection and enrichment pipeline for integrating external corporate and credit-related information into internal analytical systems. The solution combined company data from YouControl with additional external data sources to support machine learning, customer acquisition, and risk-analysis use cases.

Customer
CEE Banking & Financial Services company

Involvement Duration
4 months

Project Role
Data Scientist / Data Engineer

Responsibilities

Designed a scalable pipeline for collecting and processing information on millions of Ukrainian companies
Implemented multiple data acquisition approaches using APIs, authenticated web requests, asynchronous processing, and controlled scraping
Developed mechanisms for request batching, retries, rate-limit handling, authentication management, and data-quality validation
Built data transformation and entity-matching logic for combining YouControl, uBKI, and internal corporate records
Optimized processing performance and evaluated alternative collection strategies based on cost, stability, and scalability
Prepared enriched datasets for downstream ML models, corporate-client analytics, acquisition, and risk-related use cases

Project Team Size
3 team members (Data Scientist + Data/Business stakeholders)

Tools & Technologies
Python, Pandas, NumPy, Requests, aiohttp, BeautifulSoup, SQL, Oracle, REST APIs, Jupyter, Git

6. Corporate Recommendation Model

Project Description
Developed a machine learning recommendation system for identifying the most relevant banking products for corporate clients. The solution analyzed historical transactions, financial indicators, and product usage to generate personalized next-best-product recommendations across liabilities, assets, foreign exchange, and commission products.

Customer
CEE Banking & Financial Services company

Involvement Duration
1 year

Project Role
Data Scientist

Responsibilities

Designed a multi-model recommendation framework for corporate banking products across liabilities, assets, FX, and commission categories
Built and evaluated more than ten supervised machine learning models for product propensity and recommendation
Engineered features from financial statements, transaction history, product usage, client behavior, and external company information
Applied logistic regression, LightGBM, XGBoost, probability calibration, and product-specific threshold optimization
Implemented cross-validation, leakage checks, validation datasets, and business-rule post-processing
Automated processing of large transactional datasets containing more than 10M records per run
Delivered prioritized next-best-product recommendations to relationship managers and worked directly with business stakeholders to validate model outputs

Project Team Size
4 team members (Data Scientist + cross-functional business stakeholders)

Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, SQL, Oracle, Power BI, Jupyter, Excel

7. Kaggle — Make Data Count: Finding Data References

Project Description
Developed an NLP/LLM pipeline for detecting dataset references in scientific publications and classifying them as Primary or Secondary data usage. The solution combined large language models with engineered contextual features and classical machine learning in a stacking ensemble, achieving a Silver Medal and 28th place in the Kaggle competition.

Customer
Kaggle / Make Data Count

Involvement Duration
3 months

Project Role
Machine Learning Engineer / Kaggle Competitor

Responsibilities

Developed an end-to-end NLP pipeline for extracting dataset identifiers and references from full-text scientific publications
Parsed scientific PDF documents and processed DOI references, accession identifiers, surrounding text, and document metadata
Applied Qwen and Gemma language models to classify dataset mentions and distinguish actual dataset references from ordinary scientific citations
Engineered contextual, linguistic, metadata, and model-output features for downstream classification
Built a stacking ensemble combining predictions from multiple LLM-based pipelines using Logistic Regression as a meta-classifier
Developed inference pipelines optimized for GPU execution under Kaggle computational constraints
Performed error analysis, model comparison, feature engineering, and ensemble optimization based on leaderboard performance
Achieved a Silver Medal and Top-30 result in a competition with more than 1,200 participating teams

Project Team Size
Individual participant

Tools & Technologies
Python, Pandas, NumPy, PyTorch, Qwen, Gemma, vLLM, Scikit-learn, Logistic Regression, spaCy, Ray, PyMuPDF, pdfminer, Regex, Jupyter, Kaggle GPU
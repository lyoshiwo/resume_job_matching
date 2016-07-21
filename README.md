# resume_job_matching
Job search through online matching engines nowadays are very prominent and beneficial to both job seekers and employers with
information directly extracted from resumes and vacancies. But the solutions of traditional engines without understanding the semantic 
meanings of different resumes have not kept pace with the incredible changes
in machine learning techniques and computing capability. These solutions are usually driven by manual search-based rules and predefined
weights of keywords which lead to an inefficient and frustrating search
experience. To this end, we present a deep learning solution with rich
features including three configurable modules that can be plugged with
little restrictions. Namely, unsurprised feature extraction, base classifiers
training and ensemble method learning. The major contributions of our
work are divided into three aspects. Rather than using manual rules, ma-
chine learned methods to automatically detected the semantic similarity
of positions are proposed. Then several competitive “shallow” estimators and “deep” estimators are selected. Finally, an ensemble algorithm
to bag these estimators and aggregate their individual predictions to
form a final prediction is verified. Experimental results over 47 thousand
resumes show that our solution can significantly improve the predication
precision of job matching, including current position, salary, educational
background (detect abnormal candidates who may fake their background
or may be really excellent) and company scale.

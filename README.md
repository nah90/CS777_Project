#Term Project
Spark History:

https://2zed7a2pevamhpz3itnphgexgi-dot-us-central1.dataproc.googleusercontent.com/gateway/default/sparkhistory/history/application_1650857930129_0001/jobs/

Incorporated pretty visualization of created data referencing code from BigDataAnalytics GitHub Repository:  'Spark-Example-20a-Sgdm-with-Tree-Aggregate.ipynb'. Uploaded by Dimitar Trajanov, code originally authored by Yi Rong (yirong@bu.edu) and Xiaoyang Wang (gnayoiax@bu.edu). 
https://github.com/trajanov/BigDataAnalytics/blob/master/Notebooks/Spark-Example-20a-Code-Optimizaton.ipynb

PowerPoint Presentation:

https://docs.google.com/presentation/d/1loM4cAY0RC0ksQhd6VXELTTz4gN4i84iCm_e7kkuSlA/edit?usp=sharing
I forgot to show References after Conclusions on video presentation, but they're there.

#SVMs/Transformation/Kernel Trick in PySpark

Began project wanting to apply kernel transformation to SVM for use in PySpark with Big Data. Started with a PySpark implementation of Linear SVM with Soft Margin and created 4 separate datasets to run classifier on- three of which would run poorly with no transformation or kernel trick.
Implemented data transformations (up to higher dimensions) into model for Figures 2, 3, and 4- greatly improving accuracy over Linear SVM. Attempted to implement Kernel Trick in PySpark.
Was not able to fit data and model well with Polynomial kernel, but was able to with Gaussian kernel. Coded two ways to create the Gram matrix necessary  for Gaussian- one with use of sklearn's pairwise_functions capability.
Noted that PySpark MLlib does not currently have capability for kernels in SVM- citing issues with distributed processing the kernel calculations as well as higher demand for other (at the time - 2019) missing/lacking machine learning algorithms in MLlib.
Noted data error warnings in Google Cloud-referring to the Gram matrix. SVM with kernel trick is a very powerful learning tool, but is probably better utilized for smaller datasets. If implemented properly with distributed processing, however, it could be very powerful.
